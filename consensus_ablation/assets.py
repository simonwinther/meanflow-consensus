"""Validated, persistent official model assets; no credentials enter manifests."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

CHECKPOINT_ID = "19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu"
CHECKPOINT_URL = f"https://drive.google.com/file/d/{CHECKPOINT_ID}/view"
VAE_ID = "pcuenq/sd-vae-ft-mse-flax"
VAE_REVISION = "7581b0a0489cc8483c876a728b830b9ce087cf03"
CHECKPOINT_SHA256 = "cfe35729ecf7dbf6aa4afc51933444a2a45886ce60a55453489e1e5c3b8ffd3c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _inspect_zip(path):
    if not zipfile.is_zipfile(path):
        with Path(path).open("rb") as stream:
            beginning = stream.read(100).lower()
        kind = "HTML login, consent, quota or error page" if b"html" in beginning else "invalid ZIP"
        raise RuntimeError(f"Official checkpoint download is an {kind}: {path}. Retry public Google Drive access or provide the authorized archive.")
    with zipfile.ZipFile(path) as archive:
        names = []
        for item in archive.infolist():
            member = PurePosixPath(item.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in item.filename:
                raise RuntimeError(f"Unsafe checkpoint archive member: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError(f"Checkpoint ZIP contains a symbolic link: {item.filename}")
            names.append(item.filename)
        if not any("checkpoint_" in name for name in names):
            raise RuntimeError("Official archive contains no Flax checkpoint_ entry")
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError(f"Checkpoint archive CRC failure: {invalid}")
        return names


def load_assets(config):
    root = Path(config.get("assets_root", config.get("paths", {}).get("assets_root", "work/meanflow-assets"))).expanduser().resolve()
    path = root / "assets.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset manifest {path}; run prepare-assets first")
    assets = json.loads(path.read_text())
    for key in ("checkpoint_dir", "vae_path"):
        if not Path(assets[key]).exists():
            raise FileNotFoundError(f"Asset {key} is missing at {assets[key]}; rerun prepare-assets")
    return assets


def prepare_assets(config):
    """Download public assets once, validate them, and pin their immutable identities."""
    root = Path(config.get("assets_root", config.get("paths", {}).get("assets_root", "work/meanflow-assets"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / "assets.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        archive = root / "b4_240ep_kappa0.5.zip"
        partial = archive.with_suffix(archive.suffix + ".partial")
        expected = config.get("checkpoint_sha256") or previous.get("checkpoint_sha256") or CHECKPOINT_SHA256
        if not archive.exists():
            # A valid completed .partial can be left by the public-download preparation stage.
            if not partial.exists() or not zipfile.is_zipfile(partial):
                try:
                    import gdown
                    result = gdown.download(id=CHECKPOINT_ID, output=str(partial), quiet=False, resume=True)
                except Exception as error:
                    raise RuntimeError(f"Official Google Drive checkpoint {CHECKPOINT_URL} could not be downloaded ({type(error).__name__}). Use supported Drive authentication or supply the authorized archive at {archive}.") from error
                if not result:
                    raise RuntimeError(f"Google Drive did not return the checkpoint. Resolve access to {CHECKPOINT_URL} or place its authorized ZIP at {archive}.")
            _inspect_zip(partial)
            checksum = sha256(partial)
            if expected and checksum != expected:
                raise RuntimeError("Downloaded official checkpoint SHA-256 differs from the pinned manifest")
            os.replace(partial, archive)
        else:
            _inspect_zip(archive)
            checksum = sha256(archive)
            if expected and checksum != expected:
                raise RuntimeError("Existing checkpoint archive SHA-256 differs from the pinned manifest")
        names = _inspect_zip(archive)
        extracted = root / "checkpoint"
        marker = extracted / ".archive_sha256"
        if not marker.is_file() or marker.read_text().strip() != checksum:
            if extracted.exists():
                raise RuntimeError(f"Existing extracted checkpoint lacks the matching validation marker: {extracted}; preserve it and use a fresh assets root")
            temporary = Path(tempfile.mkdtemp(prefix="checkpoint.partial.", dir=root))
            try:
                with zipfile.ZipFile(archive) as zipped:
                    zipped.extractall(temporary)
                (temporary / ".archive_sha256").write_text(checksum + "\n")
                os.replace(temporary, extracted)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        checkpoint_entries = sorted(extracted.rglob("checkpoint_*"))
        checkpoint_entries = [path for path in checkpoint_entries if path.is_dir() or path.is_file()]
        if not checkpoint_entries:
            raise RuntimeError("Validated archive extracted without a checkpoint entry")
        # Pass the directory containing checkpoint_<step> to the official Flax loader.
        checkpoint_dir = checkpoint_entries[0].parent
        from huggingface_hub import snapshot_download
        vae_id = config.get("vae_id", VAE_ID)
        if vae_id != VAE_ID:
            raise ValueError(f"This protocol requires exact VAE {VAE_ID}")
        revision = config.get("vae_revision") or previous.get("vae_revision") or VAE_REVISION
        try:
            snapshot = Path(snapshot_download(repo_id=vae_id, revision=revision,
                cache_dir=str(root / "huggingface"), allow_patterns=["config.json", "diffusion_flax_model.msgpack"]))
        except Exception as error:
            raise RuntimeError(f"Cannot resolve exact VAE {vae_id} revision {revision or 'current commit'} ({type(error).__name__}). Authenticate through the supported Hugging Face login only if this exact asset requires access, or supply its authorized local snapshot.") from error
        required = [snapshot / "config.json", snapshot / "diffusion_flax_model.msgpack"]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Incomplete exact VAE snapshot; missing {path.name}")
        vae_files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in required}
        result = {"schema_version": 1, "checkpoint_source": CHECKPOINT_URL,
            "checkpoint_archive": str(archive), "checkpoint_bytes": archive.stat().st_size,
            "checkpoint_sha256": checksum, "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_archive_members": names, "vae_id": vae_id, "vae_revision": revision,
            "vae_path": str(snapshot), "vae_files": vae_files, "vae_latent_scale": 0.18215,
            "vae_dtype": "float32", "image_conversion": "uint8(clip(127.5*raw+128,0,255))"}
        _atomic_json(manifest_path, result)
        return result
