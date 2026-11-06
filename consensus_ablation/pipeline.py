"""Pool-atomic generation, aggregation, feature caching, and bounded execution."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import tarfile
import numpy as np
from PIL import Image
from .config import pools,pool_name
from .storage import atomic_json,atomic_npz,read_json,load_npz_validated,is_complete_npz
from .candidates import pool_candidate_ids,pool_candidate_keys
from .aggregation import image_mean,paired_latent_mean,to_uint8
from .clustering import prepare_gram,fit_kmeans

METHOD_A='image_cluster_image_mean'
METHOD_B='velocity_cluster_latent_mean_decode'
DIAGNOSTIC='velocity_cluster_image_mean'
STOP=False

class GpuTelemetry:
    def __init__(self):
        self.peak_gib=0.;self.stop=threading.Event();self.thread=threading.Thread(target=self.collect,daemon=True)
    def collect(self):
        while not self.stop.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                self.peak_gib=max(self.peak_gib,float(value.strip())/1024)
            except (OSError,ValueError,subprocess.CalledProcessError):pass
            self.stop.wait(1.)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=3)

def _signal(signum, frame):
    global STOP
    STOP=True
    print(f'Received signal {signum}; finishing current pool before stopping',flush=True)

def source_hash():
    root=Path(__file__).resolve().parents[1]; digest=hashlib.sha256()
    files=sorted((root/'consensus_ablation').glob('*.py'))+sorted((root/'models').glob('*.py'))+[root/'meanflow.py']
    for f in files:
        digest.update(str(f.relative_to(root)).encode());digest.update(f.read_bytes())
    return digest.hexdigest()

def resolve_run(c):
    from .assets import load_assets
    from .model_adapter import configure_numerics_environment
    assets=load_assets(c)
    numerical_environment=configure_numerics_environment(c)
    c['cache_identity']=dict(config_hash=c['config_hash'],upstream_commit=c['upstream_commit'],
        implementation_sha256=source_hash(),checkpoint_sha256=assets['checkpoint_sha256'],
        vae_revision=assets['vae_revision'],vae_weights_sha256=assets['vae_files']['diffusion_flax_model.msgpack']['sha256'],
        precision='float32',numerics=c['numerics'],xla_flags=numerical_environment['xla_flags'],
        jax_allocator=numerical_environment['allocator'],
        cublas_workspace_config=numerical_environment['cublas_workspace_config'],latent_scale=0.18215)
    root=Path(c['output_dir']);root.mkdir(parents=True,exist_ok=True)
    path=root/'resolved_config.json'
    if path.exists() and read_json(path).get('cache_identity')!=c['cache_identity']:
        raise ValueError(f'Run identity changed; preserve {root} and choose a new run_id')
    atomic_json(path,c)
    atomic_json(root/'class_manifest.json',dict(seed=1729,classes=c['classes'],mapping=c['class_mapping'],mapping_sha256=c['class_mapping_sha256']))
    source=Path(__file__).resolve().parents[1]
    if not (root/'source_snapshot.tar.gz').exists():
        with tarfile.open(root/'source_snapshot.tar.gz.partial','w:gz') as archive:
            for name in ['consensus_ablation','models','meanflow.py','configs','scripts/consensus','README.md','LICENSE']:
                archive.add(source/name,arcname=name,filter=lambda t:None if '__pycache__' in t.name else t)
        os.replace(root/'source_snapshot.tar.gz.partial',root/'source_snapshot.tar.gz')
        def git(*args):
            result=subprocess.run(['git',*args],cwd=source,capture_output=True,text=True)
            return result.stdout if result.returncode==0 else None
        atomic_json(root/'source_provenance.json',dict(upstream_url='https://github.com/Gsunshine/meanflow',
            upstream_commit=c['upstream_commit'],implementation_commit=git('rev-parse','HEAD'),
            working_tree_status=git('status','--short'),diff=git('diff','HEAD'),implementation_sha256=c['cache_identity']['implementation_sha256']))
        info=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],capture_output=True,text=True)
        (root/'gpu_identity.txt').write_text(info.stdout)
        for lock in ['environment-inference.lock.txt','environment-features.lock.txt']:
            original=root.parent.parent/lock
            if original.exists():(root/lock).write_bytes(original.read_bytes())
    return c

def require_allocation(c):
    if c.get('execution',{}).get('backend') == 'local': return
    if os.environ.get('SLURM_JOB_ID'): return
    path=Path('/work/JobParameters.json')
    if c.get('execution',{}).get('backend')=='ucloud' and path.exists():
        p=read_json(path)['request']; product=p['product']['id']
        if product=='gpu-nvidia-b200-1-gpu' and p['replicas']==1:
            return
    raise RuntimeError('Set execution.backend to local for an assigned GPU, or use a Slurm/UCloud allocation')

def feature_stage(c,paths=None,reference=False):
    command=[c['paths']['feature_python'],'-m','consensus_ablation.features','--config-json',str(Path(c['output_dir'])/'resolved_config.json')]
    if reference:command+=['--reference']
    command += [str(p) for p in paths or []]
    subprocess.run(command,check=True)

def _noise(model,p,n):
    keys=np.asarray(pool_candidate_keys(count=n,**p),dtype=np.uint32)
    jax=model.jax
    fn=jax.jit(jax.vmap(lambda key:jax.random.normal(key,(32,32,4),dtype=model.jnp.float32)))
    return np.asarray(jax.device_get(fn(keys)),dtype=np.float32)

def process_pool(c,model,p,telemetry=None):
    root=Path(c['output_dir']);name=pool_name(p);begin=time.perf_counter();n=c['candidates_per_pool']
    cold_start=not getattr(model,'_pool_seen',False)
    model._pool_seen=True
    identity={**c['cache_identity'],'pool':p}
    candidate_path=root/'candidates'/p['split']/(name+'.npz')
    ids=pool_candidate_ids(count=n,**p)
    gen_seconds=0.0
    if candidate_path.exists():
        cache=load_npz_validated(candidate_path,identity,required_arrays=['epsilon','velocity','candidate_ids'])
        epsilon,velocity=cache['epsilon'],cache['velocity']
        if cache['candidate_ids'].tolist()!=ids: raise ValueError('Candidate ID mismatch')
    else:
        tic=time.perf_counter();epsilon=_noise(model,p,n);velocity,endpoints=model.generate(epsilon,p['class_id'])
        gen_seconds=time.perf_counter()-tic
        np.testing.assert_allclose(endpoints,epsilon-velocity,rtol=0,atol=0)
        atomic_npz(candidate_path,identity=identity,epsilon=epsilon,velocity=velocity,candidate_ids=np.asarray(ids))
    endpoints=epsilon-velocity
    tic=time.perf_counter();images=model.decode(endpoints);candidate_decode_seconds=time.perf_counter()-tic
    assert images.shape==(n,256,256,3)
    outputs=[to_uint8(images[0])]; metadata=[];records=[];latents=[];latent_positions=[]
    base=dict(**p,run_id=c['run_id'],preset=c['preset'],config_hash=c['config_hash'],candidate_cache_key=name,
        candidate_count=n,total_candidate_evaluations=n)
    def row(method,k,seed,fit):
        size=len(fit['selected_indices'])
        return dict(**base,method=method,k=k,kmeans_seed=seed,selected_cluster_size=size,
            selected_cluster_fraction=size/n,selected_candidate_indices=fit['selected_indices'],selected_cluster_dispersion=fit.get('selected_cluster_dispersion',0.),
            full_pool_dispersion=fit.get('full_pool_dispersion'),winner_tie_count=fit.get('winner_tie_count'),converged=fit.get('converged'),
            generator_seconds=0.,decoder_seconds=0.,clustering_seconds=0.,feature_seconds=0.,total_seconds=0.,peak_gpu_memory_gib=None)
    metadata.append(row('single_sample',None,None,dict(selected_indices=[0])))
    clustering_seconds=0.;tic=time.perf_counter()
    grams={'image':prepare_gram(images),'velocity':prepare_gram(velocity)}
    clustering_seconds+=time.perf_counter()-tic
    fits={}
    for representation,samples in [('image',images),('velocity',velocity)]:
        for k in c['k_values']:
            for seed in ([None] if k==1 else c['kmeans_seeds']):
                tic=time.perf_counter()
                fit=fit_kmeans(samples,k,0 if seed is None else seed,n_init=c['n_init'],max_iter=c['max_iter'],candidate_ids=ids,gram=grams[representation])
                elapsed=time.perf_counter()-tic;clustering_seconds+=elapsed
                record={**base,**fit,'clustering_representation':representation,'elapsed_seconds':elapsed}
                records.append(record);fits[(representation,k,seed)]=fit
                selected=fit['selected_indices']
                if representation=='image':
                    outputs.append(to_uint8(image_mean(images,selected)));metadata.append(row(METHOD_A,k,seed,fit))
                else:
                    latent=paired_latent_mean(epsilon,velocity,selected,epsilon_ids=ids,velocity_ids=ids)
                    np.testing.assert_allclose(latent,np.mean(endpoints[selected].astype(np.float64),axis=0),rtol=2e-5,atol=2e-5)
                    latents.append(latent);latent_positions.append(len(outputs));outputs.append(None);metadata.append(row(METHOD_B,k,seed,fit))
                    outputs.append(to_uint8(image_mean(images,selected)));metadata.append(row(DIAGNOSTIC,k,seed,fit))
    tic=time.perf_counter();decoded=model.decode(np.stack(latents));aggregate_decode_seconds=time.perf_counter()-tic
    for pos,im in zip(latent_positions,decoded):outputs[pos]=to_uint8(im)
    diagnostic=[]
    for index,pos in enumerate(latent_positions):
        ids_=fits[('velocity',metadata[pos]['k'],metadata[pos]['kmeans_seed'])]['selected_indices']
        raw_mean=image_mean(images,ids_)
        diagnostic.append(dict(k=metadata[pos]['k'],kmeans_seed=metadata[pos]['kmeans_seed'],
            mean_absolute_raw_decoder_difference=float(np.abs(decoded[index]-raw_mean).mean())))
    elapsed=time.perf_counter()-begin
    timings=dict(**p,generator_seconds=gen_seconds,decoder_seconds=candidate_decode_seconds+aggregate_decode_seconds,
        candidate_decoder_seconds=candidate_decode_seconds,aggregate_decoder_seconds=aggregate_decode_seconds,
        clustering_seconds=clustering_seconds,total_seconds=elapsed,candidate_evaluations=n,outputs=len(outputs),cold_start=cold_start,
        peak_gpu_memory_gib=telemetry.peak_gib if telemetry else None)
    for r in metadata:
        for key in ['generator_seconds','decoder_seconds','clustering_seconds','total_seconds']:
            r[key]=timings[key]/len(outputs)
        r['peak_gpu_memory_gib']=timings['peak_gpu_memory_gib']
    atomic_json(root/'memberships'/p['split']/(name+'.json'),records)
    atomic_json(root/'timings'/(name+'.json'),timings)
    atomic_json(root/'decoder_diagnostic'/(name+'.json'),diagnostic)
    output_path=root/'pending_outputs'/(name+'.npz')
    atomic_npz(output_path,identity=c['cache_identity'],images=np.stack(outputs),metadata=np.asarray(json.dumps(metadata)))
    # Predetermined audit: first class, first generation seed, group zero.
    if p['class_id']==c['classes'][0] and p['generation_seed']==c['generation_seeds'][0] and p['group_id']==0:
        audit=root/'audit'/name;audit.mkdir(parents=True,exist_ok=True)
        for index,image in enumerate(images): Image.fromarray(to_uint8(image)).save(audit/f'candidate-{index:02d}.png')
        for index,(image,r) in enumerate(zip(outputs,metadata)):
            Image.fromarray(image).save(audit/f"output-{index:02d}-{r['method']}-k{r['k']}-s{r['kmeans_seed']}.png")
        atomic_json(audit/'outputs.json',metadata)
        atomic_npz(root/'audit'/(name+'.npz'),identity=c['cache_identity'],candidate_images=to_uint8(images),outputs=np.stack(outputs),metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps(dict(pool=name,**timings)),flush=True)
    return output_path,timings

def run_stage(c,split,limit=None,stop_after=None):
    require_allocation(c);c=resolve_run(c);root=Path(c['output_dir'])
    from .numerical_audit import validate_compilation_cache
    validate_compilation_cache(c)
    signal.signal(signal.SIGTERM,_signal);signal.signal(signal.SIGINT,_signal);signal.signal(signal.SIGUSR1,_signal)
    if split=='test' and not (root/'results'/'selected_k.json').exists():
        raise ValueError('Test generation requires frozen development-selected K artifact')
    if split=='test':
        from .evaluate import validate_frozen_selection
        validate_frozen_selection(c)
    start=time.perf_counter(); ledger=root.parent/'gpu_budget.json'
    previous=read_json(ledger) if ledger.exists() else {'charged_seconds':0.}
    budget=c['gpu_hour_limit']*3600-previous['charged_seconds']
    if budget<=0: raise RuntimeError('Campaign 48 GPU-hour execution limit exhausted')
    with (root.parent/'.gpu-worker.lock').open('a+') as lock, GpuTelemetry() as telemetry:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .model_adapter import MeanFlowAdapter
        model=None;timings=[];pending=[];processed=0;initialization=0.
        try:
            for i,p in enumerate(pools(c,split)):
                if limit is not None and i>=limit:break
                feature=root/'features'/split/(pool_name(p)+'.npz')
                if is_complete_npz(feature,c['cache_identity'],required_arrays=['features','metadata']):continue
                output=root/'pending_outputs'/(pool_name(p)+'.npz')
                if not is_complete_npz(output,c['cache_identity'],required_arrays=['images','metadata']):
                    if model is None:
                        tic=time.perf_counter();model=MeanFlowAdapter(c);initialization=time.perf_counter()-tic
                        atomic_json(root/'model_provenance.json',model.provenance)
                    output,timing=process_pool(c,model,p,telemetry);timings.append(timing)
                    atomic_json(root/'sampling_validation.json',model.validation)
                pending.append(output);processed+=1
                if len(pending)>=8 or STOP or (stop_after and processed>=stop_after):
                    feature_stage(c,pending);pending=[]
                charged=time.perf_counter()-start
                atomic_json(ledger,dict(charged_seconds=previous['charged_seconds']+charged,last_run_id=c['run_id']))
                if charged>budget-300:raise RuntimeError('Stopping at safe pool boundary before execution budget expires')
                if STOP or (stop_after and processed>=stop_after):break
            if pending:feature_stage(c,pending)
        finally:
            charged=time.perf_counter()-start
            atomic_json(ledger,dict(charged_seconds=previous['charged_seconds']+charged,last_run_id=c['run_id']))
            atomic_json(root/f'execution-{split}-{time.time_ns()}.json',dict(seconds=charged,initialization_seconds=initialization,
                allocation_id=c.get('execution',{}).get('allocation_id'),slurm_job_id=os.environ.get('SLURM_JOB_ID'),processed_pools=processed))
        if STOP:raise SystemExit(128+signal.SIGTERM)
        return timings
