#!/usr/bin/env python3
"""Declarative Liara infrastructure: diff a manifest against the live account, then converge it.

  liara_apply.py plan  [-f infra/liara.yaml] [--app NAME] [--json] [--no-color]
  liara_apply.py apply [-f infra/liara.yaml] [--app NAME] [--auto-approve] [--allow-delete]

What is managed (per app): existence, hardware plan, feature bundle, private network, zero-downtime,
read-only root filesystem, running state, disks (create / grow; shrink is impossible on Liara), required
env var NAMES. Values never appear in the manifest or in plan output: a missing variable is filled at
apply time from the environment variable `LIARA_ENV__<KEY>` (i.e. from your secret store) or the apply
stops. Health checks, ports and mounts are per-release settings and are applied by infra/liara_deploy.py.

Guarantees
  * `plan` is read-only (GET requests only) and exits 0 = no changes, 2 = changes pending, 1 = invalid
    (e.g. the manifest asks for a feature the bundle does not include).
  * `apply` re-plans, prints the plan and asks for the literal answer `yes` unless --auto-approve.
  * Nothing is destroyed unless --allow-delete is given (apps only with team.exclusive: true).
  * Every step is idempotent: re-running apply after a partial failure converges.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liara_api import Liara, LiaraError, bundle_caps, hourly_price, load_manifest  # noqa: E402


@dataclass
class Change:
    op: str  # create | update | destroy | noop-warn
    app: str
    attr: str
    have: object = None
    want: object = None
    call: tuple | None = None  # (method, path, body) executed by apply
    note: str = ""


@dataclass
class Plan:
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def actionable(self):
        return [c for c in self.changes if c.op in ("create", "update", "destroy")]


class Color:
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s


def fmt(v):
    return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else f'"{v}"'


# -- planning -----------------------------------------------------------------------------
def build_plan(li: Liara, manifest: dict, only: str | None = None, allow_delete: bool = False) -> Plan:
    P = Plan()
    team = manifest.get("team") or {}
    me = li.me()
    live = {p["project_id"]: p for p in li.projects()}
    nets = {}
    try:
        nets = {n["name"]: n["_id"] for n in li.get("/v1/networks").get("networks", [])}
    except LiaraError as e:
        P.warnings.append(f"could not list networks ({e.status})")
    want_apps = [a for a in manifest.get("apps", []) if not only or a["name"] == only]
    if only and not want_apps:
        P.errors.append(f"app {only!r} is not declared in the manifest")
    cost_now = cost_want = 0.0

    for a in want_apps:
        name = a["name"]
        plan, bundle = a["plan"], a.get("bundle", "basic")
        caps = bundle_caps(me, plan, bundle)
        if not caps:
            P.errors.append(f"{name}: unknown plan/bundle {plan}/{bundle}")
        if a.get("zeroDowntime") and caps and not caps.get("zeroDowntime"):
            P.errors.append(f"{name}: zeroDowntime: true needs a bundle with zero-downtime ({bundle} has none; use standard or pro)")
        for d in a.get("disks", []):
            nb = (d.get("backup") or {}).get("native")
            if nb and caps and not caps.get("availableBackupInterval"):
                P.errors.append(f"{name}: disk {d['name']} asks for native backups, bundle {bundle} has none")
        net = a.get("network") or team.get("network")
        cost_want += hourly_price(me, plan, bundle)

        p = live.get(name)
        if not p:
            body = {"name": name, "planID": plan, "platform": a.get("platform", "docker"), "bundlePlanID": bundle}
            if net:
                if net not in nets:
                    P.errors.append(f"{name}: network {net!r} does not exist")
                body["network"] = nets.get(net, f"<id of {net}>")
            P.changes.append(Change("create", name, "app", None, f"{plan}/{bundle}", ("POST", "/v1/projects", body)))
            for d in a.get("disks", []):
                P.changes.append(Change("create", name, f"disk.{d['name']}", None, f"{d['sizeGB']} GB",
                                        ("POST", f"/v1/projects/{name}/disks", {"name": d["name"], "size": str(d["sizeGB"])})))
            for k in a.get("envs", {}).get("required", []):
                P.changes.append(Change("create", name, f"env.{k}", None, "(from secret store)", ("ENV", name, k)))
            continue

        full = li.project(name)
        pid = full.get("_id") or p.get("_id")
        cost_now += hourly_price(me, full.get("planID", ""), full.get("bundlePlanID", ""))
        if full.get("planID") != plan:
            P.changes.append(Change("update", name, "plan", full.get("planID"), plan,
                                    ("POST", f"/v1/projects/{name}/resize", {"planID": plan}), "~20 s of 502 while resizing"))
        if full.get("bundlePlanID") != bundle:
            P.changes.append(Change("update", name, "bundle", full.get("bundlePlanID"), bundle,
                                    ("POST", f"/v1/projects/{name}/resize-bundle-plan", {"bundlePlanID": bundle})))
        have_net = ((full.get("network") or p.get("network") or {}) or {}).get("name")
        if net and have_net != net:
            P.errors.append(f"{name}: network is {have_net!r}, manifest says {net!r} (Liara cannot move an app between networks; recreate it)")
        for key, attr, route in (("zeroDowntime", "zeroDowntime", "zero-downtime"),
                                 ("readOnlyRootFilesystem", "readOnlyRootFilesystem", "read-only")):
            if key in a and bool(full.get(attr)) != bool(a[key]):
                verb = "enable" if a[key] else "disable"
                P.changes.append(Change("update", name, key, bool(full.get(attr)), bool(a[key]),
                                        ("POST", f"/v1/projects/{pid}/{route}/{verb}", None)))
        want_scale = 1 if a.get("running", True) else 0
        if int(full.get("scale", 1)) != want_scale:
            P.changes.append(Change("update", name, "scale", full.get("scale"), want_scale,
                                    ("POST", f"/v1/projects/{name}/actions/scale", {"scale": want_scale})))

        live_disks = {d["name"]: d for d in li.disks(pid).get("disks", [])}
        for d in a.get("disks", []):
            have = live_disks.get(d["name"])
            if not have:
                P.changes.append(Change("create", name, f"disk.{d['name']}", None, f"{d['sizeGB']} GB",
                                        ("POST", f"/v1/projects/{name}/disks", {"name": d["name"], "size": str(d["sizeGB"])})))
            elif float(have.get("size", 0)) < float(d["sizeGB"]):
                P.changes.append(Change("update", name, f"disk.{d['name']}.size", f"{have['size']} GB", f"{d['sizeGB']} GB",
                                        ("POST", f"/v1/projects/{name}/disks/{d['name']}/resize", {"size": str(d["sizeGB"])})))
            elif float(have.get("size", 0)) > float(d["sizeGB"]):
                P.warnings.append(f"{name}: disk {d['name']} is {have['size']} GB, manifest says {d['sizeGB']} GB (disks cannot shrink; update the manifest)")
        for extra in sorted(set(live_disks) - {d["name"] for d in a.get("disks", [])}):
            if allow_delete:
                P.changes.append(Change("destroy", name, f"disk.{extra}", f"{live_disks[extra].get('size')} GB", None,
                                        ("DELETE", f"/v1/projects/{pid}/disks/{extra}", None), "DATA LOSS"))
            else:
                P.warnings.append(f"{name}: disk {extra} exists but is not declared (kept; --allow-delete would remove it)")

        have_env = {e["key"] for e in full.get("envs", [])}
        req = a.get("envs", {}).get("required", [])
        for k in req:
            if k not in have_env:
                P.changes.append(Change("create", name, f"env.{k}", None, "(from secret store)", ("ENV", name, k)))
        unmanaged = sorted(have_env - set(req) - set(a.get("envs", {}).get("optional", [])))
        if unmanaged:
            P.info.append(f"{name}: {len(unmanaged)} env var(s) not declared in the manifest: {', '.join(unmanaged)}")

        if not caps.get("zeroDowntime"):
            P.info.append(f"{name}: bundle {bundle} has no zero-downtime; the health check still blocks bad releases, "
                          f"but each deploy has a short cutover gap")
        if not caps.get("availableBackupInterval") and a.get("disks"):
            P.info.append(f"{name}: bundle {bundle} has no native disk backups; covered by the off-platform export (ops/backup)")
        sc = a.get("scaling") or {}
        if sc:
            P.info.append(f"{name}: scaling policy {sc.get('min')}..{sc.get('max')} (vertical; enforced by Negahban/runbook, Liara has no autoscaling API)")

    if team.get("exclusive") and not only:
        declared = {a["name"] for a in manifest.get("apps", [])}
        for extra in sorted(set(live) - declared):
            if allow_delete:
                P.changes.append(Change("destroy", extra, "app", live[extra].get("planID"), None,
                                        ("DELETE", f"/v1/projects/{extra}", None), "DELETES THE APP AND ITS DISKS"))
            else:
                P.warnings.append(f"app {extra} exists on the account but is not declared (kept; --allow-delete would remove it)")

    budget = team.get("budget") or {}
    try:
        bal = float((me.get("team") or {}).get("balance") or li.get("/v1/billing")["user"]["balance"])
    except (LiaraError, KeyError, TypeError, ValueError):
        bal = None
    if not only:
        P.info.append(f"cost: {cost_now:,.0f} -> {cost_want:,.0f} toman/hour for declared apps "
                      f"(~{cost_want * 720 / 1e6:.2f}M toman / 30 days)")
        if bal is not None and cost_want:
            runway = bal / max(cost_want, 1)
            msg = f"balance {bal:,.0f} toman = {runway:,.0f} h ({runway / 24:.1f} days) of runway at the declared run-rate"
            floor = budget.get("minRunwayHours")
            (P.warnings if floor and runway < floor else P.info).append(msg + (f" (floor {floor} h)" if floor else ""))
    return P


def render(P: Plan, color: bool) -> str:
    c = Color(color)
    sym = {"create": c("32", "+"), "update": c("33", "~"), "destroy": c("31", "-")}
    out, by_app = [], {}
    for ch in P.actionable:
        by_app.setdefault(ch.app, []).append(ch)
    for app, chs in by_app.items():
        head = "create" if any(x.attr == "app" and x.op == "create" for x in chs) else (
            "destroy" if any(x.attr == "app" and x.op == "destroy" for x in chs) else "update")
        out.append(f"  {sym[head]} liara_app.{app.replace('-', '_')} {{")
        for ch in chs:
            if ch.attr == "app":
                continue
            arrow = f"{fmt(ch.have)} -> {fmt(ch.want)}" if ch.op == "update" else fmt(ch.want if ch.op == "create" else ch.have)
            out.append(f"      {sym[ch.op]} {ch.attr:<24} = {arrow}" + (c("2", f"   # {ch.note}") if ch.note else ""))
        out.append("    }")
    n = {k: sum(1 for x in P.actionable if x.op == k) for k in ("create", "update", "destroy")}
    for e in P.errors:
        out.append(c("31", f"Error: {e}"))
    for w in P.warnings:
        out.append(c("33", f"Warning: {w}"))
    for i in P.info:
        out.append(c("2", f"Note: {i}"))
    if not P.actionable:
        out.append(c("32", "No changes. Liara infrastructure matches the configuration."))
    else:
        out.append(c("1", f"Plan: {n['create']} to add, {n['update']} to change, {n['destroy']} to destroy."))
    return "\n".join(out)


# -- apply ------------------------------------------------------------------------------------
def execute(li: Liara, P: Plan) -> int:
    failed = 0
    for ch in P.actionable:
        method, path, body = ch.call
        label = f"{ch.op} {ch.app} {ch.attr}"
        try:
            if method == "ENV":
                _, app, key = ch.call
                val = os.getenv(f"LIARA_ENV__{key}")
                if val is None:
                    raise RuntimeError(f"set LIARA_ENV__{key} (from your secret store) to create it")
                cur = [{"key": e["key"], "value": e["value"]} for e in li.project(app).get("envs", [])]
                li.post("/v1/projects/update-envs", {"project": app, "variables": cur + [{"key": key, "value": val}]})
            else:
                li.request(method, path, body=body)
            print(f"  done  {label}")
        except (LiaraError, RuntimeError) as e:
            failed += 1
            print(f"  FAIL  {label}: {e}")
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "apply"])
    ap.add_argument("-f", "--file", default="infra/liara.yaml")
    ap.add_argument("--app")
    ap.add_argument("--json", action="store_true", help="machine-readable plan")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--allow-delete", action="store_true")
    a = ap.parse_args(argv)
    manifest = load_manifest(a.file)
    li = Liara()
    P = build_plan(li, manifest, a.app, a.allow_delete)
    color = not a.no_color and sys.stdout.isatty() and not os.getenv("NO_COLOR")
    if a.json:
        print(json.dumps({"changes": [{k: v for k, v in vars(c).items() if k != "call"} for c in P.actionable],
                          "errors": P.errors, "warnings": P.warnings, "info": P.info}, ensure_ascii=False, indent=1, default=str))
    else:
        print(f"Liara plan for {a.file}" + (f" (app {a.app})" if a.app else "") + "\n")
        print(render(P, color))
    if P.errors:
        return 1
    if a.cmd == "plan":
        return 2 if P.actionable else 0
    if not P.actionable:
        return 0
    if not a.auto_approve:
        if not sys.stdin.isatty():
            print("refusing to apply without a terminal; pass --auto-approve in automation")
            return 1
        if input("\nDo you want to perform these actions? Only 'yes' will be accepted: ").strip() != "yes":
            print("Apply cancelled.")
            return 1
    failed = execute(li, P)
    print(f"\nApply complete: {len(P.actionable) - failed} done, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
