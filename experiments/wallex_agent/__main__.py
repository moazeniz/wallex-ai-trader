"""Bounded research runs and a restart-safe paper-only service entry point."""
from pathlib import Path
import argparse,fcntl,json,os,signal,sys,threading,time
from datetime import datetime
from .runner import AgentRunner,ROOT
from .storage import atomic_json
from .monitor import ProtectiveMonitor


def summarize(report):
    return {'mode':report['mode'],'live_execution_enabled':report['live_execution_enabled'],
        'started_at':report.get('started_at'),'observed_at':report.get('observed_at'),'finished_at':report.get('finished_at'),
        'source_errors':report['source_errors'],'news_refresh':report['news_refresh'],'opportunity_ranking':report.get('opportunity_ranking',[]),
        'decisions':[{'symbol':x['symbol'],'action':x['proposal']['action'],'reasons':x['proposal']['reason'],
                      'forecast_gross':(x['quantitative_forecast'] or {}).get('predicted_gross_return'),
                      'analysts':[{k:a.get(k) for k in ['analyst_role','status','proposed_action','confidence']} for a in (x['council'] or {}).get('assessments',[])],
                      'paper_fill':x['execution'].get('event',{}).get('fill'),'duplicate':x['execution'].get('duplicate',False)} for x in report['decisions']],
        'portfolio':{k:v for k,v in report['portfolio'].items() if k!='fills'},'audit':report['audit'],'llm_budget':report['llm_budget']}


def serve(agent,poll_seconds,stop_event=None):
    """Keep the same durable portfolio through transient cycle failures and restarts."""
    stop_event=stop_event or threading.Event()
    health={'mode':'paper','live_execution_enabled':False,'pid':os.getpid(),'started_at':time.time(),
            'cycles_completed':0,'consecutive_errors':0,'last_completed_at':None,'last_error_at':None,'last_error_type':None}
    path=agent.state/'daemon.json'
    atomic_json(path,health)
    while not stop_event.is_set():
        started=time.monotonic()
        try:
            report=agent.cycle()
            health.update(cycles_completed=health['cycles_completed']+1,consecutive_errors=0,last_completed_at=time.time())
            print(json.dumps({'event':'paper_cycle_completed',**summarize(report)}),flush=True)
        except Exception as exc:
            # Arbitrary provider exceptions can contain secret-bearing HTTP objects; log only the type.
            health.update(consecutive_errors=health['consecutive_errors']+1,last_error_at=time.time(),last_error_type=type(exc).__name__)
            print(json.dumps({'event':'paper_cycle_failed','error_type':type(exc).__name__,
                              'consecutive_errors':health['consecutive_errors'],'live_execution_enabled':False}),flush=True)
        delay=max(1,poll_seconds-(time.monotonic()-started))
        health['next_cycle_not_before']=time.time()+delay
        atomic_json(path,health)
        stop_event.wait(delay)
    health['stopped_at']=time.time();atomic_json(path,health)


def supervise(agent,poll_seconds,protection_seconds=30):
    stopped=threading.Event()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda signum,frame:stopped.set())
    monitor=ProtectiveMonitor(agent)
    def protect():
        while not stopped.is_set():
            started=time.monotonic()
            try:
                report=monitor.tick()
                atomic_json(agent.state/'protection.json',report)
                if report.get('events') or report.get('errors'):
                    print(json.dumps({'event':'paper_protection',**report}),flush=True)
            except Exception as exc:
                print(json.dumps({'event':'paper_protection_failed','error_type':type(exc).__name__,
                                  'live_execution_enabled':False}),flush=True)
            stopped.wait(max(1,protection_seconds-(time.monotonic()-started)))
    worker=threading.Thread(target=protect,name='paper-protection',daemon=True)
    worker.start()
    try:serve(agent,poll_seconds,stopped)
    finally:
        stopped.set();worker.join(timeout=65)


def main():
    parser=argparse.ArgumentParser(description='Wallex multisource Haiku research council and shared paper trader; no real orders')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['run','serve']:
        run=sub.add_parser(command)
        run.add_argument('--model',required=True);run.add_argument('--state',default=str(ROOT/'.wallex-agent'))
        run.add_argument('--artifacts',default=str(ROOT/'artifacts/wallex_agent/forward-v1'))
        run.add_argument('--poll-seconds',type=int,default=300);run.add_argument('--research-limit',type=int,default=3);run.add_argument('--cutoff-utc')
        if command=='run':
            run.add_argument('--cycles',type=int,default=1);run.add_argument('--cached-news',action='store_true');run.add_argument('--symbols',nargs='+')
    status=sub.add_parser('status');status.add_argument('--state',default=str(ROOT/'.wallex-agent'))
    args=parser.parse_args()
    if args.command=='status':
        path=Path(args.state)/'status.json'
        if not path.exists():raise ValueError('No completed agent cycle')
        report=json.loads(path.read_text());output=summarize(report)
        for name in ['daemon','protection']:
            health=Path(args.state)/(name+'.json')
            if health.exists():output[name]=json.loads(health.read_text())
        print(json.dumps(output,indent=2));return
    if args.command=='run' and not 1<=args.cycles<=12:raise ValueError('Run1..12cycles explicitly; use serve for persistent paper operation')
    if args.poll_seconds<30:raise ValueError('Minimum poll interval30seconds')
    state=Path(args.state);state.mkdir(parents=True,exist_ok=True)
    with (state/'runner.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another runner owns this state directory')
        cutoff=None
        if args.cutoff_utc:
            parsed=datetime.fromisoformat(args.cutoff_utc.replace('Z','+00:00'))
            if parsed.tzinfo is None:raise ValueError('Cutoff requires timezone')
            cutoff=parsed.timestamp()
        agent=AgentRunner(args.model,args.state,args.artifacts,research_limit=args.research_limit,trading_cutoff_epoch=cutoff)
        if args.command=='serve':
            supervise(agent,args.poll_seconds)
        else:
            for i in range(args.cycles):
                report=agent.cycle(symbols=args.symbols,refresh_news=not args.cached_news);print(json.dumps(summarize(report),indent=2),flush=True)
                if i+1<args.cycles:time.sleep(args.poll_seconds)

if __name__=='__main__':
    try:main()
    except Exception as exc:print(type(exc).__name__+': '+str(exc)[:250],file=sys.stderr);sys.exit(1)
