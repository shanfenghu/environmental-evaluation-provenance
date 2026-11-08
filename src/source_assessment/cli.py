"""Command-line entry points with explicit configuration and output destinations."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Source-aware retrospective assessment reproduction')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['fit','summarize','verify','render','compare','prepare','acquire','inventory']:
        p=sub.add_parser(name)
        if name=='inventory':
            p.add_argument('--root',default='.')
            p.add_argument('--code-only',action='store_true')
        if name in ['fit','summarize','verify']:
            p.add_argument('--config',default='configs/assessment-v1.json')
        if name in ['fit','verify']:p.add_argument('--data',default='data/prepared')
        if name=='fit':
            p.add_argument('--cohorts',default='configs/cohorts-v1.json')
            p.add_argument('--full-only',action='store_true',help='Replay only the four full fitted states')
            p.add_argument('--fixed-cycle',action='store_true',help='Replay only original cycle 0 of every task')
        if name in ['summarize','verify','render','compare']:p.add_argument('--records',default='records/reference')
        if name=='render':
            p.add_argument('--pairing')
            p.add_argument('--profiles')
        if name=='compare':p.add_argument('--replay',required=True)
        if name in ['fit','summarize','verify','render','compare','prepare']:p.add_argument('--output',required=True)
        if name in ['prepare','acquire']:
            p.add_argument('--raw',required=True)
            p.add_argument('--acquisition',default='configs/acquisition-v1.json')
            p.add_argument('--allow-container-repack',action='store_true')
        if name=='prepare':p.add_argument('--preparation',default='configs/preparation-v1.json')
    args=parser.parse_args(argv)
    if args.command=='inventory':
        from .artifact import verify_inventory
        print(json.dumps(verify_inventory(args.root,require_data=not args.code_only)))
    elif args.command=='fit':
        from .driver import run
        run(args.config,args.data,args.cohorts,args.output,args.fixed_cycle,args.full_only)
    elif args.command=='summarize':
        from .summary import summarize
        summarize(args.records,args.config,args.output)
    elif args.command=='verify':
        from .verify import verify
        print(json.dumps(verify(args.records,args.data,args.config,args.output)))
    elif args.command=='render':
        from .render import render
        render(args.records,args.output)
        if args.pairing or args.profiles:
            from .tables import render_tables
            render_tables(args.records,Path(args.output)/'tables',args.pairing,args.profiles)
    elif args.command=='compare':
        from .verify import compare
        print(json.dumps(compare(args.records,args.replay,args.output)))
    elif args.command=='prepare':
        from .prepare import prepare
        prepare(args.raw,args.preparation,args.acquisition,args.output,args.allow_container_repack)
    elif args.command=='acquire':
        from .prepare import acquire
        print(json.dumps(acquire(args.raw,json.loads(Path(args.acquisition).read_text()),args.allow_container_repack)))
