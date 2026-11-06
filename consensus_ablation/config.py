"""Resolved, hashed protocols; no secrets or data files belong in source control."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml

UPSTREAM = 'd70cb55d298ee03c53bf6da67bec281082e4e2d9'

def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    c.setdefault('upstream_commit', UPSTREAM)
    c.setdefault('microbatch_size', 16)
    c.setdefault('n_init', 10)
    c.setdefault('max_iter', 100)
    c.setdefault('generation_seeds', [2026, 2027, 2028])
    c.setdefault('kmeans_seeds', [11, 29, 47])
    c.setdefault('k_values', [1, 2, 5, 10, 15])
    c.setdefault('candidates_per_pool', 64)
    c.setdefault('storage_limit_gib', 100)
    c.setdefault('gpu_hour_limit', 48)
    c.setdefault('campaign_id', 'consensus-fp32-cached-v4')
    c.setdefault('numerics', {'matmul_precision':'highest','xla_gpu_autotune_level':4,
        'xla_gpu_deterministic_ops':False,'xla_gpu_exclude_nondeterministic_ops':False,
        'persistent_compilation_cache':True})
    c.setdefault('paths', {})
    storage = Path(os.environ.get('MEANFLOW_STORAGE', 'work')).expanduser().resolve()
    c['paths'].setdefault('assets_root', str(storage / 'meanflow-assets'))
    c['paths'].setdefault('compilation_cache', str(Path(c['paths']['assets_root'])/'jax-fp32-cached-v3'))
    c['paths'].setdefault('reference_root', str(storage / 'imagenet' / 'val'))
    c['paths'].setdefault('feature_python', str(storage / 'meanflow-features-env' / 'bin' / 'python'))
    c.setdefault('output_dir', str(storage / 'meanflow-runs' / c['run_id']))
    mapping_path = Path(__file__).parent / 'imagenet_class_index.json'
    c['class_mapping'] = json.loads(mapping_path.read_text())
    c['class_mapping_sha256'] = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    main_classes = np.random.default_rng(1729).choice(1000,100,replace=False).tolist()
    c['classes'] = main_classes[:c['class_count']]
    c.setdefault('metrics', {'kid_subset_size':500, 'kid_subsets':100, 'seed':31415})
    assert c['preset'] in ('main','compact','smoke','benchmark')
    assert max(c['k_values']) <= c['candidates_per_pool']
    assert c['n_init'] == 10 and c['max_iter'] == 100
    assert c['upstream_commit'] == UPSTREAM
    c['config_hash'] = hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    return c

def pools(c, split):
    for seed in c['generation_seeds']:
        for label in c['classes']:
            for group in range(c['pools_per_class'][split]):
                yield dict(split=split,generation_seed=seed,class_id=label,group_id=group)

def pool_name(p):
    return f"{p['split']}-s{p['generation_seed']}-c{p['class_id']:04d}-g{p['group_id']:04d}"

def counts(c):
    npools = len(c['classes'])*len(c['generation_seeds'])*sum(c['pools_per_class'].values())
    configurations = 1 + 3*(1+(len(c['k_values'])-1)*len(c['kmeans_seeds']))
    return dict(pools=npools,candidate_evaluations=npools*c['candidates_per_pool'],
                outputs=npools*configurations,outputs_per_pool=configurations,
                compact_candidate_gib=npools*c['candidates_per_pool']*4096*8/2**30,
                feature_gib=npools*configurations*2048*4/2**30)
