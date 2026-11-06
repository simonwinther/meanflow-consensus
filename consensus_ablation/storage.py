"""Atomic local/persistent artifacts and strict cache-identity checks."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


class CacheValidationError(ValueError):
    pass


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False, default=_json_default)


def config_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _atomic_stream(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    with _atomic_stream(path, "w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                default=_json_default) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def atomic_npz(path, arrays=None, *, identity=None, **named_arrays):
    """Write only complete shards. Identity must include asset/config hashes.

    Arrays may be supplied as one mapping or as named arguments. An object
    dtype is never allowed, so subsequent reads need no pickle capability.
    """
    data = dict(arrays or {})
    overlap = data.keys() & named_arrays.keys()
    if overlap:
        raise ValueError(f"Duplicate NPZ array names: {sorted(overlap)}")
    data.update(named_arrays)
    if "_metadata" in data:
        raise ValueError("_metadata is a reserved cache-array name")
    data = {key: np.asarray(value) for key, value in data.items()}
    if any(array.dtype.hasobject for array in data.values()):
        raise ValueError("Object arrays are prohibited in cache shards")
    metadata = {"schema_version": 1, "complete": True, "identity": identity,
                "identity_hash": config_hash(identity)}
    data["_metadata"] = np.asarray(canonical_json(metadata))
    with _atomic_stream(path, "wb") as stream:
        np.savez_compressed(stream, **data)


def load_npz_validated(path, expected_identity, *, required_arrays=()):
    path = Path(path)
    if path.suffix != ".npz" or path.name.startswith("."):
        raise CacheValidationError("Temporary/incomplete shards cannot be reused")
    try:
        with open(path, "rb") as stream, np.load(stream, allow_pickle=False) as archive:
            if "_metadata" not in archive:
                raise CacheValidationError("Shard has no completion/identity metadata")
            metadata = json.loads(str(archive["_metadata"].item()))
            if not isinstance(metadata, dict):
                raise CacheValidationError("Shard metadata must be a JSON object")
            if metadata.get("schema_version") != 1 or metadata.get("complete") is not True:
                raise CacheValidationError("Shard schema or completion marker is invalid")
            identity = metadata.get("identity")
            if (metadata.get("identity_hash") != config_hash(identity)
                    or config_hash(identity) != config_hash(expected_identity)):
                raise CacheValidationError("Checkpoint/VAE/config cache identity changed")
            missing = set(required_arrays) - set(archive.files)
            if missing:
                raise CacheValidationError(f"Shard lacks required arrays: {sorted(missing)}")
            return {key: archive[key].copy() for key in archive.files if key != "_metadata"}
    except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        if isinstance(error, CacheValidationError):
            raise
        raise CacheValidationError(f"Invalid/incomplete cache shard {path}: {error}") from error


def is_complete_npz(path, expected_identity, *, required_arrays=()):
    try:
        load_npz_validated(path, expected_identity, required_arrays=required_arrays)
    except CacheValidationError:
        return False
    return True
