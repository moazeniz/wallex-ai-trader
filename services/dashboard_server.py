"""Password-protected HTTPS server for the Wallex arena dashboard (read-only static files).

Run by systemd with credentials: tls_cert, tls_key (Let's Encrypt files) and password (plain text, user is fixed).
Usage: dashboard_server.py PORT DIRECTORY
"""
import base64
import functools
import hmac
import os
import ssl
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CREDS = Path(os.environ["CREDENTIALS_DIRECTORY"])
USER = "meowmeow"
PASSWORD = (CREDS / "password").read_text().strip()
EXPECTED = "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()


class Handler(SimpleHTTPRequestHandler):
    server_version = "wallex-arena"
    sys_version = ""

    def _authorized(self) -> bool:
        return hmac.compare_digest(self.headers.get("Authorization", ""), EXPECTED)

    def do_GET(self):
        if not self._authorized():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Wallex Arena", charset="UTF-8"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()

    def do_HEAD(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        super().do_HEAD()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Strict-Transport-Security", "max-age=86400")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # no access log with client addresses


def main():
    port, directory = int(sys.argv[1]), sys.argv[2]
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(CREDS / "tls_cert", CREDS / "tls_key")
    server = ThreadingHTTPServer(("0.0.0.0", port), functools.partial(Handler, directory=directory))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
