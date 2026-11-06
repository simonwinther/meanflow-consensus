import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import signal
import numpy as np
from .config import load_config,counts,pools
from .storage import atomic_json,read_json,load_npz_validated

CAMPAIGN_DEADLINE=None

def reference(c):
    from .pipeline import resolve_run,require_allocation,feature_stage
    require_allocation(c);resolve_run(c)
    target=Path(c['output_dir'])/'reference'/'features.npz'
    if target.exists():
        from .reference import build_reference_manifest
        manifest=read_json(build_reference_manifest(c))
        data=load_npz_validated(target,c['cache_identity'],required_arrays=['development','test','manifest_hash','feature_provenance'])
        if data['manifest_hash'].item()!=manifest['manifest_hash']:raise ValueError('Reference feature manifest mismatch')
    else:feature_stage(c,reference=True)

def stage_subprocess(command,config):
    c=load_config(config);ledger=Path(c['output_dir']).parent/'gpu_budget.json'
    before=read_json(ledger).get('charged_seconds',0.) if ledger.exists() else 0.
    start=time.perf_counter();interrupted=[]
    child=subprocess.Popen([sys.executable,'-m','consensus_ablation',command,'--config',str(config)])
    def forward(signum,frame):
        interrupted.append(signum)
        if child.poll() is None:child.send_signal(signum)
    previous={sig:signal.signal(sig,forward) for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
    try:
        timeout=max(1.,CAMPAIGN_DEADLINE-time.monotonic()) if CAMPAIGN_DEADLINE else None
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('Campaign execution budget reached; stopped current child safely where possible')
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        current=read_json(ledger).get('charged_seconds',0.) if ledger.exists() else before
        atomic_json(ledger,dict(charged_seconds=before+max(current-before,time.perf_counter()-start),last_run_id=c['run_id']))
    if interrupted:raise SystemExit(128+interrupted[0])
    if code:raise subprocess.CalledProcessError(code,child.args)

def benchmark(c,config_path):
    from .pipeline import run_stage,resolve_run
    resolve_run(c)
    existing=Path(c['output_dir'])/'benchmark.json'
    if existing.exists():return read_json(existing)
    start=time.perf_counter();run_stage(c,'development');elapsed=time.perf_counter()-start
    root=Path(c['output_dir'])
    records=sorted([read_json(p) for p in (root/'timings').glob('*.json')],key=lambda r:(r['generation_seed'],r['class_id'],r['group_id']))
    executions=[read_json(p) for p in root.glob('execution-development-*.json')]
    # Per-pool core timing excludes first cold generation/decoder compile.
    steady=[r['total_seconds'] for r in records if not r.get('cold_start',False)]
    feature_seconds=0.
    for path in (root/'features'/'development').glob('*.npz'):
        with np.load(path,allow_pickle=False) as z:
            feature_seconds+=sum(x.get('feature_seconds',0.) for x in json.loads(z['metadata'].item()))
    initialization=sum(x.get('initialization_seconds',0.) for x in executions)
    total_execution=sum(x['seconds'] for x in executions)
    core=np.mean(steady) if steady else records[0]['total_seconds']
    # Complete measured execution/pool is an upper estimate including feature startup and cold compilation.
    full=total_execution/max(1,len(records))
    summary=dict(measured_pools=len(records),steady_core_seconds_per_pool=float(core),
        complete_seconds_per_pool=float(full),feature_seconds=feature_seconds,initialization_seconds=initialization,
        total_execution_seconds=total_execution,safety_factor=1.3,
        main_projected_seconds=float(full*7500*1.3),compact_projected_seconds=float(full*720*1.3),
        compilation_policy='first pool excluded from steady core; included conservatively in campaign projection')
    atomic_json(root/'benchmark.json',summary);print(json.dumps(summary,indent=2));return summary

def campaign(config_path):
    global CAMPAIGN_DEADLINE
    main_path=Path(config_path).resolve()
    base=main_path.parent
    c=load_config(main_path)
    if c['preset'] != 'main':raise ValueError('Campaign requires the main config; compact is selected only by the benchmark')
    runs=Path(c['output_dir']).parent
    spent=read_json(runs/'gpu_budget.json').get('charged_seconds',0.) if (runs/'gpu_budget.json').exists() else 0.
    CAMPAIGN_DEADLINE=time.monotonic()+48*3600-spent
    stage_subprocess('numerical-audit',main_path)
    # Subprocess boundaries release JAX/CUDA allocations between protocol stages.
    smoke_path=base/'consensus_smoke.yaml';bench_path=base/'consensus_benchmark.yaml'
    stage_subprocess('smoke',smoke_path)
    stage_subprocess('benchmark',bench_path)
    bench=read_json(Path(load_config(bench_path)['output_dir'])/'benchmark.json')
    runs=Path(load_config(bench_path)['output_dir']).parent
    budget=read_json(runs/'gpu_budget.json').get('charged_seconds',0.) if (runs/'gpu_budget.json').exists() else 0.
    decision_path=runs/(c['campaign_id']+'-decision.json')
    if decision_path.exists():
        decision=read_json(decision_path);chosen=Path(decision['config'])
        if decision['config_hash']!=load_config(chosen)['config_hash'] or decision['benchmark']!=bench:
            raise ValueError('Frozen campaign preset/config/benchmark changed; preserve previous run and investigate')
    elif bench['main_projected_seconds'] < 48*3600-budget:chosen=main_path
    elif bench['compact_projected_seconds'] < 48*3600-budget:chosen=base/'consensus_compact.yaml'
    else:raise RuntimeError('Measured compact preset exceeds remaining 48 GPU-hour budget; expansion needs user decision')
    c=load_config(chosen)
    estimate=counts(c);estimated_gib=estimate['compact_candidate_gib']+estimate['feature_gib']+15
    if estimated_gib>100:raise RuntimeError('Projected persistent storage exceeds 100 GiB')
    if not decision_path.exists():atomic_json(decision_path,dict(config=str(chosen),config_hash=c['config_hash'],preset=c['preset'],benchmark=bench,
        projected_storage_gib=estimated_gib,counts=estimate,
        hypotheses=['H1 largest image-cluster mean may improve over all-image mean',
        'H2 velocity clustering may change fidelity-coverage tradeoff',
        'H3 K effects depend on selected size','H4 nonlinear decode order may explain differences']))
    stage_subprocess('prepare-reference',chosen)
    for command in ['development','score-development','select-k','test','score-test','summarize']:
        stage_subprocess(command,chosen)

def main():
    parser=argparse.ArgumentParser(description='Frozen MeanFlow inference-only consensus ablation')
    parser.add_argument('command',choices=['inspect','prepare-assets','prepare-data','prepare-reference','numerical-audit','smoke','benchmark','run-stage','development','test','score','score-development','score-test','select-k','summarize','status','campaign'])
    parser.add_argument('--config',default='configs/consensus_main.yaml');parser.add_argument('--split',choices=['development','test'],default='development')
    parser.add_argument('--limit',type=int);parser.add_argument('--stop-after',type=int)
    parser.add_argument('--recorded-run',type=Path,help='Read an archived resolved config and metrics for summarize only')
    parser.add_argument('--report-dir',type=Path,help='Write summarize outputs to a separate directory')
    args=parser.parse_args()
    if (args.recorded_run or args.report_dir) and args.command!='summarize':
        parser.error('--recorded-run and --report-dir are only valid for summarize')
    if args.recorded_run:
        if not args.report_dir:parser.error('--recorded-run requires --report-dir to preserve archived results')
        c=read_json(args.recorded_run/'resolved_config.json')
        c['output_dir']=str(args.recorded_run.resolve())
    else:c=load_config(args.config)
    if args.command=='inspect':print(json.dumps(dict(config=c,counts=counts(c)),indent=2));return
    if args.command=='prepare-assets':
        from .assets import prepare_assets
        print(json.dumps(prepare_assets(c),indent=2));return
    if args.command=='prepare-data':
        from .data_assets import prepare_imagenet,download_imagenet
        download_imagenet(c)
        print(json.dumps(prepare_imagenet(c),indent=2));return
    if args.command=='campaign':campaign(args.config);return
    if args.command=='numerical-audit':
        from .numerical_audit import run_numerical_audit
        print(run_numerical_audit(c));return
    if args.command=='prepare-reference':reference(c);return
    if args.command in ['run-stage','smoke','development','test']:
        from .pipeline import run_stage
        if args.command=='smoke' and c['preset']!='smoke':raise ValueError('Use smoke preset for smoke command')
        run_stage(c,'test' if args.command=='test' else args.split,limit=args.limit,stop_after=args.stop_after);return
    if args.command=='benchmark':benchmark(c,args.config);return
    if args.command in ['score','score-development','score-test']:
        from .pipeline import resolve_run
        resolve_run(c)
        from .evaluate import evaluate_run
        split='test' if args.command=='score-test' else args.split
        print(evaluate_run(c,split));return
    if args.command=='select-k':
        from .pipeline import resolve_run
        resolve_run(c)
        from .evaluate import select_k
        print(select_k(c));return
    if args.command=='summarize':
        from .report import generate_report
        print(generate_report(c,destination=args.report_dir));return
    if args.command=='status':
        root=Path(c['output_dir'])
        print(json.dumps(dict(run_id=c['run_id'],preset=c['preset'],
            completed_features={s:len(list((root/'features'/s).glob('*.npz'))) for s in ['development','test']},
            expected_pools={s:len(list(pools(c,s))) for s in ['development','test']},
            selected_k=read_json(root/'results'/'selected_k.json') if (root/'results'/'selected_k.json').exists() else None),indent=2))

if __name__=='__main__':main()
