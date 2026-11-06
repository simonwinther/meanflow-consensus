"""Mandatory two-process replay gate using the predetermined first main pool."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .storage import atomic_json,atomic_npz,read_json,load_npz_validated,sha256_file

def validate_compilation_cache(c):
    """A resumed stage must use the already audited executable cache."""
    path=Path(c['output_dir']).parent/(c['campaign_id']+'-numerical-audit')/'passed.json'
    if not path.exists():raise RuntimeError('Run numerical-audit before releasing GPU experiment stages')
    report=read_json(path)
    expected={k:v for k,v in c['cache_identity'].items() if k!='config_hash'}
    actual={k:v for k,v in report['cache_identity'].items() if k!='config_hash'}
    if expected!=actual or report['microbatch_size']!=c['microbatch_size']:
        raise ValueError('Audited model/source/precision policy differs from this stage')
    if Path(report['compilation_cache_dir']).resolve()!=Path(c['paths']['compilation_cache']).resolve():
        raise ValueError('Executable cache path differs from the audited cache')
    if not report.get('compiled_cache_sha256'):raise ValueError('No audited compiled executables recorded')
    for name,checksum in report['compiled_cache_sha256'].items():
        if not Path(name).is_file() or sha256_file(name)!=checksum:
            raise ValueError(f'Audited executable cache changed or disappeared: {name}')
    return report

def _export(raw):
    return np.clip(np.float32(127.5)*raw+np.float32(128),0,255).astype(np.uint8)

def _direct_lloyd(x,initial):
    x=x.reshape(len(x),-1).astype(np.float64);centers=x[initial].copy();last=None
    for _ in range(100):
        distances=np.stack([np.square(x-center).sum(1) for center in centers],axis=1)
        labels=distances.argmin(1)
        if min(np.bincount(labels,minlength=len(initial)))==0:
            raise AssertionError('Independent audit fixture encountered an empty cluster')
        centers=np.stack([x[labels==j].mean(0) for j in range(len(initial))])
        if last is not None and np.array_equal(labels,last):return labels
        last=labels.copy()
    raise AssertionError('Independent Lloyd audit did not converge')

def _partition(labels):
    return sorted(tuple(np.where(np.asarray(labels)==j)[0]) for j in set(labels))

def child(config_path,destination):
    from .pipeline import process_pool,feature_stage,pool_name
    from .model_adapter import MeanFlowAdapter
    c=read_json(config_path);c['output_dir']=str(destination)
    destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/'resolved_config.json',c)
    p=dict(split='development',generation_seed=c['generation_seeds'][0],class_id=c['classes'][0],group_id=0)
    name=pool_name(p)
    model=MeanFlowAdapter(c)
    output,_=process_pool(c,model,p)
    z=load_npz_validated(destination/'candidates/development'/(name+'.npz'),{**c['cache_identity'],'pool':p})
    saved=load_npz_validated(output,c['cache_identity'])
    fits=read_json(destination/'memberships/development'/(name+'.json'))
    meta=json.loads(saved['metadata'].item());eps,u=z['epsilon'],z['velocity']
    repeat,_=model.generate(eps,p['class_id'])
    np.testing.assert_array_equal(repeat,u,err_msg='Same executable changed velocity')
    raw=model.decode(eps-u)
    fitmap={(r['clustering_representation'],r['k'],r['kmeans_seed']):r for r in fits}
    reconstructed=[None]*len(meta);latents=[];positions=[];identity_error=0.
    for i,r in enumerate(meta):
        if r['method']=='single_sample':reconstructed[i]=_export(raw[0]);continue
        representation='image' if r['method']=='image_cluster_image_mean' else 'velocity'
        members=fitmap[representation,r['k'],r['kmeans_seed']]['selected_indices']
        if r['method']=='velocity_cluster_latent_mean_decode':
            latent=eps[members].astype(np.float64).mean(0)-u[members].astype(np.float64).mean(0)
            endpoint_mean=(eps-u)[members].astype(np.float64).mean(0)
            identity_error=max(identity_error,float(np.max(np.abs(latent-endpoint_mean))))
            np.testing.assert_allclose(latent,endpoint_mean,rtol=2e-5,atol=2e-5)
            latents.append(latent.astype(np.float32));positions.append(i)
        else:reconstructed[i]=_export(raw[members].mean(0,dtype=np.float64).astype(np.float32))
    for i,image in zip(positions,model.decode(np.stack(latents))):reconstructed[i]=_export(image)
    np.testing.assert_array_equal(np.stack(reconstructed),saved['images'],err_msg='Independent aggregation changed final output')
    max_inertia_relative_error=0.
    for r in fits:
        flat=(raw if r['clustering_representation']=='image' else u).reshape(64,-1).astype(np.float64)
        labels=np.asarray(r['labels']);centers=np.stack([flat[labels==j].mean(0) for j in range(r['k'])])
        residual=np.square(flat-centers[labels]).sum(1)
        inertia=float(residual.sum())
        rel=abs(inertia-r['inertia'])/max(1,inertia);max_inertia_relative_error=max(max_inertia_relative_error,rel)
        if rel>1e-10:raise AssertionError('Full-coordinate inertia differs from optimized Gram calculation')
        if r['k']>1 and r['inertia']!=min(r['restart_inertias']):raise AssertionError('Incorrect restart selection')
        counts=np.bincount(labels,minlength=r['k']);possible=np.where(counts==max(counts))[0]
        dispersion=np.asarray([residual[labels==j].mean() for j in range(r['k'])])
        low=dispersion[possible].min();possible=[j for j in possible if np.isclose(dispersion[j],low,rtol=1e-10,atol=1e-10)]
        ids=z['candidate_ids'].tolist()
        winner=min(possible,key=lambda j:tuple(sorted(ids[v] for v in np.where(labels==j)[0])))
        if np.where(labels==winner)[0].tolist()!=r['selected_indices']:raise AssertionError('Independent largest-cluster selection differs')
    for representation,k,seed in [('image',5,29),('velocity',15,47)]:
        r=fitmap[representation,k,seed]
        actual=_direct_lloyd(raw if representation=='image' else u,r['initial_center_indices'])
        if _partition(actual.tolist())!=_partition(r['labels']):raise AssertionError('Independent direct Lloyd partition differs')
    atomic_npz(destination/'replay.npz',identity=c['cache_identity'],epsilon=eps,velocity=u,raw_images=raw,outputs=saved['images'])
    atomic_json(destination/'checks.json',dict(status='passed',all_40_independent_aggregates_bitwise_equal=True,
        paired_latent_identity_max_abs=identity_error,all_26_direct_inertias_max_relative_error=max_inertia_relative_error,
        direct_lloyd_representatives=2,model_provenance=model.provenance))
    feature_stage(c,[output])

def run_numerical_audit(c):
    from .pipeline import require_allocation,resolve_run
    require_allocation(c);resolve_run(c)
    root=Path(c['output_dir']).parent/(c['campaign_id']+'-numerical-audit')
    root.mkdir(parents=True,exist_ok=True)
    report_path=root/'passed.json'
    if report_path.exists():
        report=read_json(report_path)
        if report['cache_identity']!=c['cache_identity']:raise ValueError('Numerical audit identity changed')
        for path,checksum in report['artifact_sha256'].items():
            if sha256_file(path)!=checksum:raise ValueError('Numerical audit artifact checksum changed')
        validate_compilation_cache(c)
        return report_path
    protocol=root/'protocol.json';atomic_json(protocol,c)
    started=time.perf_counter()
    with (root.parent/'.gpu-worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for replica in ['replica-a','replica-b']:
            with (root/(replica+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-m','consensus_ablation.numerical_audit','--child',str(protocol),'--destination',str(root/replica)],stdout=log,stderr=subprocess.STDOUT,check=True)
    a=load_npz_validated(root/'replica-a/replay.npz',c['cache_identity'])
    b=load_npz_validated(root/'replica-b/replay.npz',c['cache_identity'])
    comparisons={}
    for key in ['epsilon','velocity','raw_images','outputs']:
        np.testing.assert_array_equal(a[key],b[key],err_msg=f'Fresh-process {key} differs')
        comparisons[key]=dict(bitwise_equal=True,max_abs=0.,shape=list(a[key].shape))
    feature_paths=[next((root/replica/'features/development').glob('*.npz')) for replica in ['replica-a','replica-b']]
    fa,fb=[load_npz_validated(path,c['cache_identity']) for path in feature_paths]
    np.testing.assert_array_equal(fa['features'],fb['features'],err_msg='Fresh-process Inception features differ')
    if fa['feature_provenance'].item()!=fb['feature_provenance'].item():raise ValueError('Feature numerical policy mismatch')
    comparisons['features']=dict(bitwise_equal=True,max_abs=0.,shape=list(fa['features'].shape))
    artifact_paths=[root/'replica-a/replay.npz',root/'replica-b/replay.npz',*feature_paths,
        root/'replica-a/checks.json',root/'replica-b/checks.json']
    cached=sorted(p for p in Path(c['paths']['compilation_cache']).rglob('*') if p.is_file() and not p.name.endswith('-atime'))
    if not cached:raise ValueError('Persistent executable cache was not populated by replay')
    atomic_json(report_path,dict(status='passed',cache_identity=c['cache_identity'],pool='first main development pool',
        fresh_process_replicas=2,candidate_count_per_replica=64,outputs_per_replica=40,
        compilation_cache_dir=c['paths']['compilation_cache'],
        microbatch_size=c['microbatch_size'],compiled_cache_sha256={str(p):sha256_file(p) for p in cached},
        comparisons=comparisons,seconds=time.perf_counter()-started,
        artifact_sha256={str(path):sha256_file(path) for path in artifact_paths}))
    return report_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--child',required=True);p.add_argument('--destination',required=True)
    args=p.parse_args();child(args.child,Path(args.destination))

if __name__=='__main__':main()
