"""Isolated PyTorch Inception feature stage; never imports JAX."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .storage import atomic_npz, atomic_json, load_npz_validated, read_json

def configure_torch_numerics():
    """Fix CUDA determinism before discovering a device or loading Inception."""
    workspace = ':4096:8'
    existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if existing is not None and existing != workspace:
        raise RuntimeError('Feature extraction requires CUBLAS_WORKSPACE_CONFIG=:4096:8; conflicting numerical policies cannot share a run')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = dict(cublas_workspace_config=workspace,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        network_dtype='float32',autocast=False,
        fixed_batch_size=32,padding='uint8_zero_images_removed_before_caching')
    return torch, policy

def center_crop(image, size=256):
    # ADM evaluation convention: repeated BOX downsampling then BICUBIC and center crop.
    image = image.convert('RGB')
    while min(image.size) >= 2*size:
        image = image.resize(tuple(x//2 for x in image.size),resample=Image.Resampling.BOX)
    scale = size/min(image.size)
    image = image.resize(tuple(round(x*scale) for x in image.size),resample=Image.Resampling.BICUBIC)
    arr = np.asarray(image)
    y,x=(arr.shape[0]-size)//2,(arr.shape[1]-size)//2
    return arr[y:y+size,x:x+size]

class Extractor:
    def __init__(self):
        torch, numerical_policy = configure_torch_numerics()
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        torch.set_num_threads(8)
        self.torch=torch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.net=FeatureExtractorInceptionV3('inception-v3-compat',['2048']).eval().to(self.device)
        weights=list((Path(torch.hub.get_dir())/'checkpoints').glob('weights-inception-2015-12-05-6726825d.pth'))
        if len(weights)!=1: raise RuntimeError('Cannot identify Inception weights for provenance')
        self.provenance=dict(implementation='torch-fidelity',version='0.3.0',
            extractor='inception-v3-compat',layer='2048',
            weights_sha256=hashlib.sha256(weights[0].read_bytes()).hexdigest(),
            real_preprocessing='adm_center_crop_256_rgb_uint8',
            generated_preprocessing='127.5*x+128_clip_uint8',
            numerical_policy=numerical_policy)

    def __call__(self,images):
        t=self.torch; outputs=[]
        with t.inference_mode():
            for start in range(0,len(images),32):
                source = images[start:start+32]
                count = len(source)
                if count < 32:
                    source = np.concatenate([source, np.zeros((32-count, *source.shape[1:]), dtype=source.dtype)])
                batch=t.from_numpy(np.ascontiguousarray(source.transpose(0,3,1,2))).to(self.device)
                outputs.append(self.net(batch)[0].cpu().numpy()[:count].astype(np.float32))
        return np.concatenate(outputs)

def extract_outputs(config, paths):
    ext=Extractor(); root=Path(config['output_dir'])
    for path in map(Path,paths):
        tic=time.perf_counter()
        z=load_npz_validated(path,config['cache_identity'],required_arrays=['images','metadata'])
        metadata=json.loads(z['metadata'].item())
        feat=ext(z['images'])
        duration=time.perf_counter()-tic
        for row in metadata: row['feature_seconds']=duration/len(metadata)
        target=root/'features'/metadata[0]['split']/path.name
        atomic_npz(target,identity=config['cache_identity'],features=feat,
            metadata=np.asarray(json.dumps(metadata)),feature_provenance=np.asarray(json.dumps(ext.provenance)))
        atomic_json(root/'completed'/path.with_suffix('.json').name,
                    dict(identity=config['cache_identity'],feature_file=str(target),output_count=len(feat)))
        path.unlink()
        print(json.dumps(dict(feature_file=str(target),seconds=duration)),flush=True)

def extract_reference(config):
    from .reference import build_reference_manifest
    manifest_path=build_reference_manifest(config)
    manifest=read_json(manifest_path)
    ext=Extractor(); arrays={}; tic=time.perf_counter()
    # Accept both records and images keys from manifest builder.
    records=manifest.get('records',manifest.get('images',[]))
    for split in ['development','test']:
        selected=[r for r in records if r['split']==split]
        features=[]
        for start in range(0,len(selected),32):
            images=np.stack([center_crop(Image.open(r['path'])) for r in selected[start:start+32]])
            features.append(ext(images))
        if not features: raise ValueError(f'No reference images for {split}; inspect manifest format')
        arrays[split]=np.concatenate(features)
    digest=manifest.get('manifest_hash',hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest())
    atomic_npz(Path(config['output_dir'])/'reference'/'features.npz',identity=config['cache_identity'],
        **arrays,manifest_hash=np.asarray(digest),feature_provenance=np.asarray(json.dumps(ext.provenance)))
    atomic_json(Path(config['output_dir'])/'reference'/'timing.json',dict(seconds=time.perf_counter()-tic))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config-json',required=True); p.add_argument('--reference',action='store_true');p.add_argument('paths',nargs='*')
    args=p.parse_args(); config=read_json(args.config_json)
    if args.reference: extract_reference(config)
    else: extract_outputs(config,args.paths)

if __name__=='__main__': main()
