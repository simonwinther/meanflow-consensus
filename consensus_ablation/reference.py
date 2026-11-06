"""Validated ImageNet validation manifests; no substitute for authorized data."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def config_dict(config):
    if isinstance(config, dict):
        return config
    for name in ("as_dict", "to_dict"):
        if hasattr(config, name):
            return getattr(config, name)()
    if hasattr(config, "raw"):
        return config.raw
    raise TypeError("Configuration must be a dictionary or expose as_dict()/to_dict()")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def reference_manifest_path(config):
    c = config_dict(config)
    return Path(c["output_dir"]) / "reference" / "manifest.json"


def build_reference_manifest(config):
    """Require exactly 50 files per selected synset and fix a 20/30 split.

    A class-local SHA-256 seed makes the main/compact prefix share the same real
    split. The algorithm and exact file hashes are persisted in the manifest.
    """
    import numpy as np

    c = config_dict(config)
    root_value = c.get("paths", {}).get("reference_root")
    if not root_value:
        raise FileNotFoundError("ImageNet validation data is required: set paths.reference_root to an authorized synset-folder dataset; full-ImageNet FID statistics cannot replace it")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory does not exist: {root}")
    mapping = c.get("class_mapping")
    if not mapping:
        raise ValueError("A verified class_mapping is required before building references")
    classes = [int(x) for x in c["classes"]]
    if len(classes) != len(set(classes)):
        raise ValueError("Duplicate selected class IDs")
    seed = int(c.get("reference_split_seed", 2718))
    dev_count, test_count = 20, 30
    entries = []
    class_manifest = []
    allowed = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
    for class_id in classes:
        item = mapping.get(str(class_id), mapping.get(class_id))
        if isinstance(item, (list, tuple)):
            synset, name = item[0], item[1] if len(item) > 1 else None
        elif isinstance(item, dict):
            synset, name = item["synset"], item.get("name")
        else:
            raise ValueError(f"Missing or invalid mapping for class {class_id}")
        if not isinstance(synset, str) or len(synset) != 9 or synset[0] != "n" or not synset[1:].isdigit():
            raise ValueError(f"Invalid ImageNet synset for class {class_id}: {synset!r}")
        directory = root / synset
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing selected ImageNet class directory: {directory}")
        images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed)
        if len(images) != dev_count + test_count:
            raise ValueError(f"Class {class_id} ({synset}) has {len(images)} validation images, expected exactly 50; resolve dataset discrepancy before evaluation")
        local_seed = int(stable_hash([seed, class_id, synset])[:16], 16)
        order = np.random.default_rng(local_seed).permutation(len(images))
        development = set(int(i) for i in order[:dev_count])
        class_manifest.append({"class_id": class_id, "synset": synset, "name": name})
        for index, path in enumerate(images):
            entries.append({"id": f"{synset}/{path.name}", "path": str(path.resolve()), "class_id": class_id,
                            "synset": synset, "split": "development" if index in development else "test",
                            "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    payload = {"schema_version": 1, "dataset": "ImageNet-1K standard validation", "root": str(root),
               "split_seed": seed, "split_algorithm": "PCG64(class-local SHA256(seed,class_id,synset) first 64 bits); sorted filenames; first 20 development",
               "development_per_class": dev_count, "test_per_class": test_count,
               "classes": class_manifest, "images": entries}
    payload["manifest_hash"] = stable_hash(payload)
    path = reference_manifest_path(c)
    if path.exists():
        old = json.loads(path.read_text())
        if old != payload:
            raise ValueError(f"Reference manifest changed for an existing run: {path}; use a new run directory")
    else:
        atomic_json(path, payload)
    return path


def validate_reference_manifest(path):
    payload = json.loads(Path(path).read_text())
    recorded = payload.pop("manifest_hash")
    if stable_hash(payload) != recorded:
        raise ValueError("Reference manifest hash mismatch")
    payload["manifest_hash"] = recorded
    identifiers = [entry["id"] for entry in payload["images"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Reference image appears more than once (development/test leakage)")
    for item in payload["classes"]:
        for split, expected in (("development", 20), ("test", 30)):
            count = sum(x["class_id"] == item["class_id"] and x["split"] == split for x in payload["images"])
            if count != expected:
                raise ValueError(f"Wrong reference count for class {item['class_id']} split {split}")
    return payload
