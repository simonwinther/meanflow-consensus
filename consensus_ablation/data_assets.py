"""Verify official ImageNet validation assets and retain only the fixed 100 classes."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

import numpy as np

from .assets import _atomic_json, sha256

# Published in torchvision's official ImageNet parser, not hashes inferred from a download.
ARCHIVES = {"ILSVRC2012_img_val.tar": "29b22e2961454d5413ddabcf34fc5622",
            "ILSVRC2012_devkit_t12.tar.gz": "fa75699e90414af021442c21a62c3abf"}
HASH_SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py"

def download_imagenet(config):
    """Fetch official public validation/devkit archives, retaining resumable partials."""
    paths=config.get('paths',{})
    root=Path(paths.get('imagenet_archive_root',Path(paths.get('reference_root','work/imagenet/val')).parent))
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.download.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name in ARCHIVES:
            if (root/name).exists():continue
            target=root/(name+'.partial')
            offset=target.stat().st_size if target.exists() else 0
            expected={'ILSVRC2012_img_val.tar':6744924160,'ILSVRC2012_devkit_t12.tar.gz':2568145}[name]
            if offset==expected:continue
            req=urllib.request.Request('https://image-net.org/data/ILSVRC/2012/'+name,
                headers={'Range':f'bytes={offset}-'} if offset else {})
            try:
                with urllib.request.urlopen(req,timeout=60) as source:
                    if 'html' in source.headers.get('Content-Type','').lower():
                        raise RuntimeError('Official ImageNet returned an HTML login/error page')
                    append=offset>0 and source.status==206
                    with target.open('ab' if append else 'wb') as output:
                        shutil.copyfileobj(source,output,length=8*1024*1024)
                        output.flush();os.fsync(output.fileno())
            except Exception as error:
                raise RuntimeError(f'Official ImageNet asset {name} download failed: {type(error).__name__}: {error}. Resolve official ImageNet access or provide the authorized archive; no substitute dataset is used.') from error
            if target.stat().st_size!=expected:raise RuntimeError(f'Incomplete download {target}; rerun prepare-data to resume')
    return root


def _hashes(path):
    md5, secure = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            secure.update(chunk)
    return md5.hexdigest(), secure.hexdigest()


def _verified_archive(directory, name):
    expected = ARCHIVES[name]
    path = directory / name
    partial = directory / (name + ".partial")
    candidate = path if path.exists() else partial
    if not candidate.is_file():
        raise FileNotFoundError(f"Required official ImageNet asset is missing: {path}. Supply the authorized archive or finish the public download.")
    before = candidate.stat()
    observed, secure = _hashes(candidate)
    after = candidate.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"Download is still being written: {candidate}; rerun preparation after the download finishes")
    if observed != expected:
        state = "unfinished or corrupt download" if candidate == partial else "corrupt or unexpected archive"
        raise RuntimeError(f"ImageNet {state}: {candidate}; MD5 {observed}, expected published MD5 {expected}. Complete the download before preparation.")
    if candidate == partial:
        os.replace(partial, path)
    return path, {"filename": name, "bytes": after.st_size, "md5": observed,
                  "sha256": secure, "published_hash_source": HASH_SOURCE}


def _safe_member(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
        raise RuntimeError(f"Unsafe ImageNet archive member: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise RuntimeError(f"Unexpected link or special file in ImageNet archive: {member.name}")


def _devkit_metadata(path):
    from scipy.io import loadmat

    metadata = ground_truth = None
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            _safe_member(member)
            if member.name.endswith("/data/meta.mat"):
                metadata = archive.extractfile(member).read()
            elif member.name.endswith("/data/ILSVRC2012_validation_ground_truth.txt"):
                ground_truth = archive.extractfile(member).read()
    if metadata is None or ground_truth is None:
        raise RuntimeError("Official devkit lacks meta.mat or validation ground-truth labels")
    synsets = loadmat(io.BytesIO(metadata), squeeze_me=True, struct_as_record=False)["synsets"]
    id_to_synset = {int(item.ILSVRC2012_ID): str(item.WNID) for item in synsets
                    if int(item.num_children) == 0}
    if len(id_to_synset) != 1000 or len(set(id_to_synset.values())) != 1000:
        raise RuntimeError("Official devkit does not define 1000 distinct ImageNet leaf classes")
    labels = [int(value) for value in ground_truth.decode("ascii").split()]
    if len(labels) != 50000 or not set(labels).issubset(id_to_synset):
        raise RuntimeError("Official devkit does not define all 50,000 validation labels")
    return id_to_synset, labels, {"meta_mat_sha256": hashlib.sha256(metadata).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(ground_truth).hexdigest()}


def prepare_imagenet(config):
    """Promote verified archives and atomically extract the main protocol's 5,000 references.

    No training data is downloaded or extracted. Ground-truth ILSVRC IDs are
    translated through meta.mat to WNIDs, then checked against the model mapping.
    Compact and smoke configurations reuse a prefix of this fixed class subset.
    """
    paths = config.get("paths", {})
    reference_root = Path(paths.get("reference_root", "work/imagenet/val")).expanduser().resolve()
    directory = Path(paths.get("imagenet_archive_root", reference_root.parent)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mapping = config.get("class_mapping")
    if not mapping or len(mapping) != 1000:
        raise ValueError("prepare_imagenet requires the complete verified 1000-class model mapping")
    model_synsets = {}
    for index in range(1000):
        item = mapping.get(str(index), mapping.get(index))
        synset = item[0] if isinstance(item, (tuple, list)) else item["synset"]
        if not re.fullmatch(r"n\d{8}", synset):
            raise ValueError(f"Invalid model synset mapping at index {index}")
        model_synsets[index] = synset
    if len(set(model_synsets.values())) != 1000:
        raise ValueError("Model synset mapping is not one-to-one")
    classes = np.random.default_rng(1729).choice(1000, 100, replace=False).tolist()
    if not set(config.get("classes", classes)).issubset(classes):
        raise ValueError("Requested classes differ from the predefined 100-class main subset")
    expected_synsets = {model_synsets[index] for index in classes}
    with (directory / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = reference_root / ".prepared.json"
        if marker.is_file():
            previous = json.loads(marker.read_text())
            if previous.get("classes") != classes or previous.get("class_mapping_sha256") != config.get("class_mapping_sha256"):
                raise RuntimeError("Existing prepared references use a different class manifest")
            for synset in expected_synsets:
                if len(list((reference_root / synset).glob("*.JPEG"))) != 50:
                    raise RuntimeError(f"Previously prepared class {synset} no longer contains 50 images")
            return previous
        val_archive, val_provenance = _verified_archive(directory, "ILSVRC2012_img_val.tar")
        devkit, devkit_provenance = _verified_archive(directory, "ILSVRC2012_devkit_t12.tar.gz")
        id_to_synset, labels, metadata_provenance = _devkit_metadata(devkit)
        if set(id_to_synset.values()) != set(model_synsets.values()):
            raise RuntimeError("Official devkit leaf WNIDs differ from the frozen model class mapping")
        if reference_root.exists():
            raise RuntimeError(f"Existing {reference_root} has no completed preparation manifest. Preserve it and choose a fresh reference_root or inspect the existing dataset.")
        reference_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=reference_root.name + ".partial.", dir=reference_root.parent))
        counts = {synset: 0 for synset in expected_synsets}
        identifiers, seen = [], set()
        try:
            for synset in expected_synsets:
                (temporary / synset).mkdir()
            with tarfile.open(val_archive, "r|") as archive:
                for member in archive:
                    _safe_member(member)
                    if member.isdir():
                        continue
                    name = PurePosixPath(member.name).name
                    match = re.fullmatch(r"ILSVRC2012_val_(\d{8})\.JPEG", name)
                    if not match:
                        raise RuntimeError(f"Unexpected official validation image name: {member.name}")
                    position = int(match.group(1))
                    if not 1 <= position <= 50000 or position in seen:
                        raise RuntimeError(f"Duplicate or invalid validation image index: {position}")
                    seen.add(position)
                    synset = id_to_synset[labels[position - 1]]
                    if synset not in expected_synsets:
                        continue
                    output = temporary / synset / name
                    with archive.extractfile(member) as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    counts[synset] += 1
                    identifiers.append({"id": f"{synset}/{name}", "bytes": member.size,
                        "sha256": sha256(output), "official_validation_index": position,
                        "ilsvrc_class_id": labels[position - 1]})
            if len(seen) != 50000 or set(counts.values()) != {50}:
                raise RuntimeError(f"Incomplete official validation archive: {len(seen)} images, selected class counts {counts}")
            result = {"schema_version": 1, "dataset": "ImageNet ILSVRC2012 validation",
                "reference_root": str(reference_root), "classes": classes,
                "class_selection_seed": 1729, "class_mapping_sha256": config.get("class_mapping_sha256"),
                "class_mapping_verified_against": "official devkit meta.mat leaf WNIDs",
                "label_translation": "validation filename index -> official ground-truth ILSVRC ID -> meta.mat WNID -> frozen zero-based model index",
                "selected_image_count": len(identifiers), "archives": [val_provenance, devkit_provenance],
                **metadata_provenance, "images": sorted(identifiers, key=lambda item: item["id"])}
            _atomic_json(temporary / ".prepared.json", result)
            os.replace(temporary, reference_root)
            return result
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
