"""Deterministic distribution metrics and development-only K selection.

KID: polynomial unbiased MMD^2, matching torch-fidelity 0.3.0's mathematical
definition (including population SD over subset draws). Precision/recall: the
union-of-kNN-balls definition of Kynkaanniemi et al. (2019), evaluated on the same
pinned Inception features as KID/FID, with self excluded and k=3 by default.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .reference import (atomic_json, config_dict, file_sha256,
                        reference_manifest_path, stable_hash,
                        validate_reference_manifest)
from .storage import load_npz_validated

IMAGE_METHOD = "image_cluster_image_mean"
VELOCITY_METHOD = "velocity_cluster_latent_mean_decode"
DIAGNOSTIC_METHOD = "velocity_cluster_image_mean"
MAIN_METHODS = (IMAGE_METHOD, VELOCITY_METHOD)
ALL_METHODS = ("single_sample", IMAGE_METHOD, VELOCITY_METHOD, DIAGNOSTIC_METHOD)
METRIC_FIELDS = [
    "run_id", "preset", "split", "method", "k", "generation_seed", "kmeans_seed",
    "generated_count", "reference_count", "kid", "kid_subset_std", "fid",
    "generative_precision", "generative_recall", "mean_selected_cluster_size",
    "mean_selected_cluster_fraction", "mean_cluster_dispersion", "mean_full_pool_dispersion",
    "winner_tie_frequency", "convergence_failure_count", "total_candidate_evaluations",
    "generator_seconds", "decoder_seconds", "clustering_seconds", "feature_seconds",
    "core_seconds", "metric_seconds", "total_seconds", "peak_gpu_memory_gib", "config_hash",
    "reference_manifest_hash", "generated_pool_manifest_hash", "feature_provenance_hash",
    "kid_schedule_hash", "kid_subset_size", "kid_subsets", "precision_recall_neighbors",
]


def atomic_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _features(values, name="features"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least two feature rows")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {name}")
    return values


def polynomial_mmd_unbiased(real, generated):
    """Exact U-statistic; excludes within-set diagonals and permits negatives."""
    x, y = _features(real), _features(generated)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    d = x.shape[1]
    xx, yy, xy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
    return float((xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
                 + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1)) - 2 * xy.mean())


def kid_subset_schedule(real_count, generated_count, subset_size=500, subsets=100, seed=31415):
    if subset_size < 2 or subset_size > min(real_count, generated_count) or subsets < 1:
        raise ValueError(f"KID requires 2 <= subset_size={subset_size} <= min(real={real_count}, generated={generated_count}) and positive subsets; do not duplicate images or silently shrink the protocol")
    rng = np.random.RandomState(int(seed))
    real, generated = [], []
    for _ in range(int(subsets)):
        real.append(rng.choice(real_count, subset_size, replace=False))
        generated.append(rng.choice(generated_count, subset_size, replace=False))
    return np.asarray(real, dtype=np.int64), np.asarray(generated, dtype=np.int64)


def kernel_inception_distance(real, generated, subset_size=500, subsets=100, seed=31415, schedule=None):
    x, y = _features(real), _features(generated)
    schedule = schedule if schedule is not None else kid_subset_schedule(len(x), len(y), subset_size, subsets, seed)
    if len(schedule) != 2 or schedule[0].shape != schedule[1].shape or schedule[0].ndim != 2:
        raise ValueError("Invalid shared KID subset schedule")
    estimates = np.asarray([polynomial_mmd_unbiased(x[i], y[j]) for i, j in zip(*schedule)])
    return {"kid": float(estimates.mean()), "kid_subset_std": float(estimates.std(ddof=0)), "kid_draws": estimates}


class ReferenceMetrics:
    """Cache reference-only statistics while preserving the exact estimators."""

    def __init__(self, real, neighbors=3):
        self.real = _features(real, "reference features")
        self.neighbors = int(neighbors)
        if self.neighbors < 1 or len(self.real) <= self.neighbors:
            raise ValueError("Precision/recall needs more observations than neighbors")
        self.mean = self.real.mean(axis=0)
        centered = self.real - self.mean
        self.covariance = centered.T @ centered / (len(self.real) - 1)
        self.trace_covariance = float(np.trace(self.covariance))
        self.radii = _knn_radii(self.real, self.neighbors)
        self.kernel = (self.real @ self.real.T / self.real.shape[1] + 1) ** 3

    def kid(self, generated, schedule):
        y = _features(generated)
        d = self.real.shape[1]
        yy = (y @ y.T / d + 1) ** 3
        xy = (self.real @ y.T / d + 1) ** 3
        estimates = []
        for i, j in zip(*schedule):
            xx_i, yy_j = self.kernel[np.ix_(i, i)], yy[np.ix_(j, j)]
            m = len(i)
            estimates.append((xx_i.sum() - np.trace(xx_i) + yy_j.sum() - np.trace(yy_j)) / (m * (m - 1))
                             - 2 * xy[np.ix_(i, j)].mean())
        return float(np.mean(estimates)), float(np.std(estimates, ddof=0))

    def fid(self, generated):
        y = _features(generated)
        if y.shape[1] != self.real.shape[1]:
            raise ValueError("Feature dimensions differ")
        mean = y.mean(axis=0)
        centered = y - mean
        trace_generated = float(np.square(centered).sum() / (len(y) - 1))
        # The nonzero eigenvalues of C_real C_generated equal those of this
        # smaller symmetric matrix, avoiding nonsymmetric square-root errors.
        if len(y) <= y.shape[1]:
            symmetric = centered @ self.covariance @ centered.T / (len(y) - 1)
        else:
            eig, vec = np.linalg.eigh(self.covariance)
            square_root = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
            generated_covariance = centered.T @ centered / (len(y) - 1)
            symmetric = square_root @ generated_covariance @ square_root
        from scipy.linalg import eigvalsh
        eigenvalues = eigvalsh((symmetric + symmetric.T) / 2, check_finite=False)
        tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues.min() < -tolerance:
            raise ValueError("Materially non-PSD FID covariance product")
        cross_trace = np.sqrt(np.maximum(eigenvalues, 0)).sum()
        value = float(np.square(self.mean - mean).sum() + self.trace_covariance + trace_generated - 2 * cross_trace)
        if value < -1e-7 * max(1.0, self.trace_covariance + trace_generated):
            raise ValueError("Materially negative FID indicates numerical failure")
        return max(value, 0.0)

    def precision_recall(self, generated):
        y = _features(generated)
        radii = _knn_radii(y, self.neighbors)
        return (_manifold_coverage(y, self.real, self.radii),
                _manifold_coverage(self.real, y, radii))


def frechet_inception_distance(real, generated):
    # Avoid the optional manifold calculations for standalone analytic tests.
    obj = object.__new__(ReferenceMetrics)
    obj.real = _features(real)
    obj.mean = obj.real.mean(axis=0)
    centered = obj.real - obj.mean
    obj.covariance = centered.T @ centered / (len(obj.real) - 1)
    obj.trace_covariance = float(np.trace(obj.covariance))
    return obj.fid(generated)


def _squared_distances(x, y):
    return np.maximum(np.square(x).sum(axis=1)[:, None] + np.square(y).sum(axis=1)[None, :] - 2 * x @ y.T, 0)


def _knn_radii(features, neighbors=3, block_size=256):
    if len(features) <= neighbors or neighbors < 1:
        raise ValueError("Precision/recall needs more observations than neighbors")
    radii = np.empty(len(features), dtype=np.float64)
    for begin in range(0, len(features), block_size):
        end = min(begin + block_size, len(features))
        distances = _squared_distances(features[begin:end], features)
        distances[np.arange(end - begin), np.arange(begin, end)] = np.inf
        radii[begin:end] = np.partition(distances, neighbors - 1, axis=1)[:, neighbors - 1]
    return radii


def _manifold_coverage(probes, centers, radii, block_size=256):
    inside = 0
    for begin in range(0, len(probes), block_size):
        distances = _squared_distances(probes[begin:begin + block_size], centers)
        inside += int(np.any(distances <= radii[None, :], axis=1).sum())
    return inside / len(probes)


def generative_precision_recall(real, generated, neighbors=3):
    x, y = _features(real), _features(generated)
    return (_manifold_coverage(y, x, _knn_radii(x, neighbors)),
            _manifold_coverage(x, y, _knn_radii(y, neighbors)))


def _json_scalar(archive, key):
    if key not in archive:
        raise ValueError(f"Feature archive is missing {key}")
    value = archive[key]
    value = value.item() if value.ndim == 0 else value.tolist()
    return json.loads(value) if isinstance(value, str) else value


def _pool_key(row):
    return int(row["class_id"]), int(row["group_id"])


def _cache_identity(c):
    path = Path(c["output_dir"]) / "resolved_config.json"
    if not path.is_file():
        raise ValueError("Scoring requires resolved_config.json with the pinned source/checkpoint/VAE cache identity")
    resolved = json.loads(path.read_text())
    identity = resolved.get("cache_identity")
    if not isinstance(identity, dict) or identity.get("config_hash") != c["config_hash"] or resolved.get("config_hash") != c["config_hash"]:
        raise ValueError("Resolved scoring/config cache identity mismatch")
    if c.get("cache_identity", identity) != identity:
        raise ValueError("Current/resolved scoring cache identity mismatch")
    return identity


def _generated_archives(c, split):
    directory = Path(c["output_dir"]) / "features" / split
    return sorted(p for p in directory.rglob("*.npz")
                  if not p.name.startswith(".") and not p.name.endswith(".tmp.npz"))


def load_generated_features(config, split):
    c = config_dict(config)
    directory = Path(c["output_dir"]) / "features" / split
    cache_identity = _cache_identity(c)
    groups, provenance, all_ids = defaultdict(list), None, set()
    archives = _generated_archives(c, split)
    if not archives:
        raise FileNotFoundError(f"No complete generated feature archives in {directory}")
    for path in archives:
        archive = load_npz_validated(path, cache_identity, required_arrays=["features", "metadata", "feature_provenance"])
        if archive:
            features = np.asarray(archive["features"])
            if features.ndim != 2 or features.shape[1] != 2048 or not np.isfinite(features).all():
                raise ValueError(f"Expected finite Inception pool3 (M,2048) features in {path}")
            metadata = _json_scalar(archive, "metadata")
            current_provenance = _json_scalar(archive, "feature_provenance")
            if provenance is not None and provenance != current_provenance:
                raise ValueError("Generated feature pipelines differ")
            provenance = current_provenance
            if len(metadata) != len(features):
                raise ValueError(f"Metadata/feature count mismatch: {path}")
            for feature, row in zip(features, metadata):
                row = dict(row)
                if row.get("split") != split:
                    raise ValueError(f"Cross-split feature leakage in {path}")
                if row.get("config_hash") != c["config_hash"]:
                    raise ValueError(f"Feature/config hash mismatch in {path}")
                if row["method"] not in ALL_METHODS:
                    raise ValueError(f"Undeclared output method: {row['method']}")
                k = row.get("k")
                if row["method"] == "single_sample":
                    k = None
                elif k is None or int(k) < 1:
                    raise ValueError("Consensus output is missing K")
                else:
                    k = int(k)
                km = row.get("kmeans_seed")
                if k in (None, 1):
                    if km is not None:
                        raise ValueError("Single-sample/K=1 rows must not imply clustering-seed repetitions")
                elif km is None:
                    raise ValueError("K>1 output is missing clustering seed")
                else:
                    km = int(km)
                group = (row["method"], k, int(row["generation_seed"]), km)
                identity = group + _pool_key(row)
                if identity in all_ids:
                    raise ValueError(f"Duplicated generated output: {identity}")
                all_ids.add(identity)
                groups[group].append((row, feature))
    populations = defaultdict(set)
    for (method, k, generation, km), rows in groups.items():
        rows.sort(key=lambda pair: _pool_key(pair[0]))
        populations[generation].add(tuple(_pool_key(pair[0]) for pair in rows))
    if any(len(population) != 1 for population in populations.values()):
        raise ValueError("Methods/K/clustering seeds have unmatched generated pool IDs; incomplete cases cannot be silently dropped")
    if len(set(next(iter(x)) for x in populations.values())) != 1:
        raise ValueError("Generation seeds do not have matching class/group populations")
    # Every declared output configuration must exist for each generation seed.
    configuration_sets = defaultdict(set)
    for method, k, generation, km in groups:
        configuration_sets[generation].add((method, k, km))
    if len({tuple(sorted(values, key=str)) for values in configuration_sets.values()}) != 1:
        raise ValueError("Missing method/K/clustering-seed configurations in a generation population")
    expected_configurations = {("single_sample", None, None)}
    for method in (IMAGE_METHOD, VELOCITY_METHOD, DIAGNOSTIC_METHOD):
        for k in c["k_values"]:
            for km in ([None] if int(k) == 1 else c["kmeans_seeds"]):
                expected_configurations.add((method, int(k), None if km is None else int(km)))
    if any(values != expected_configurations for values in configuration_sets.values()):
        raise ValueError("Generated features do not contain the exact declared method/K/clustering-seed grid")
    declared_seeds = c.get("generation_seeds")
    if declared_seeds and set(map(int, declared_seeds)) != set(configuration_sets):
        raise ValueError("Feature cache is missing declared generation seeds")
    pools_per_class = c.get(f"{split}_pools_per_class", c.get("pools_per_class", {}).get(split) if isinstance(c.get("pools_per_class"), dict) else None)
    if pools_per_class is not None:
        expected = {(int(cid), group) for cid in c["classes"] for group in range(int(pools_per_class))}
        for population in populations.values():
            if set(next(iter(population))) != expected:
                raise ValueError("Feature population does not match the preregistered classes/groups")
    return groups, provenance


def _metric_settings(c):
    metrics = c.get("metrics", {})
    default_size = {"main": 500, "compact": 40, "smoke": 4}.get(c.get("preset"), 500)
    return {"subset_size": int(metrics.get("kid_subset_size", default_size)),
            "subsets": int(metrics.get("kid_subsets", 100)),
            "seed": int(metrics.get("seed", 31415)),
            "neighbors": int(metrics.get("precision_recall_neighbors", 3))}


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _sum(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.sum(values)) if values else None


def _runtime_fields(metadata, metric_seconds):
    # Older shards called the amortized generation/aggregation core "total".
    core = [row.get("core_seconds", row.get("total_seconds")) for row in metadata]
    features = [row.get("feature_seconds") for row in metadata]
    core_seconds = float(sum(core)) if core and all(x is not None for x in core) else None
    feature_seconds = float(sum(features)) if features and all(x is not None for x in features) else None
    total = (core_seconds + feature_seconds + metric_seconds
             if core_seconds is not None and feature_seconds is not None else None)
    return dict(core_seconds=core_seconds, feature_seconds=feature_seconds,
                metric_seconds=metric_seconds, total_seconds=total)


def _selection_guard(c, manifest_hash, provenance_hash):
    path = Path(c["output_dir"]) / "results" / "selected_k.json"
    if not path.exists():
        raise RuntimeError("Test metrics are gated on frozen development selection; run select-k first")
    selection = json.loads(path.read_text())
    expected = {"config_hash": c["config_hash"], "reference_manifest_hash": manifest_hash,
                "feature_provenance_hash": provenance_hash, "metric_settings": _metric_settings(c)}
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Frozen selection mismatch: {key}")
    development = path.parent / "development_metrics.csv"
    if not development.exists() or file_sha256(development) != selection["development_metrics_sha256"]:
        raise ValueError("Frozen development scores changed or disappeared")
    return selection


def _reference_context(c):
    reference_path = Path(c.get("paths", {}).get("reference_features") or Path(c["output_dir"]) / "reference" / "features.npz")
    if not reference_path.is_file() or not reference_manifest_path(c).is_file():
        raise FileNotFoundError("Real ImageNet subset features/manifest unavailable; no distribution metrics or K selection computed. Synthetic smoke outputs do not measure model quality.")
    manifest = validate_reference_manifest(reference_manifest_path(c))
    archive = load_npz_validated(reference_path, _cache_identity(c), required_arrays=["development", "test", "manifest_hash", "feature_provenance"])
    manifest_hash = str(archive["manifest_hash"].item())
    provenance = _json_scalar(archive, "feature_provenance")
    if manifest_hash != manifest["manifest_hash"]:
        raise ValueError("Reference feature manifest hash mismatch")
    if [x["class_id"] for x in manifest["classes"]] != list(map(int, c["classes"])):
        raise ValueError("Reference classes/order differ from run configuration")
    for split in ("development", "test"):
        real = _features(archive[split], "reference features")
        expected = sum(row["split"] == split for row in manifest["images"])
        if real.shape[1] != 2048 or len(real) != expected:
            raise ValueError("Reference feature count/dimension does not match split manifest")
    return reference_path, manifest, archive, provenance


def _evaluation_fingerprint(c, split, reference_path, manifest_hash, provenance):
    root = Path(c["output_dir"])
    return {"schema_version": 2, "config_hash": c["config_hash"], "cache_identity": _cache_identity(c),
            "reference_manifest_hash": manifest_hash, "feature_provenance_hash": stable_hash(provenance),
            "metric_settings": _metric_settings(c), "reference_features_sha256": file_sha256(reference_path),
            "generated_features_manifest_hash": stable_hash([
                [str(path.relative_to(root)), file_sha256(path)] for path in _generated_archives(c, split)])}


def _completed_metrics(c, split, fingerprint, *, frozen=False):
    root = Path(c["output_dir"]) / "results"
    path, status_path = root / f"{split}_metrics.csv", root / f"{split}_evaluation_status.json"
    if path.is_file() and status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("status") == "complete" and status.get("input_fingerprint") == fingerprint and status.get("metrics_csv_sha256") == file_sha256(path):
            return path
    if frozen:
        raise ValueError("Frozen development metrics lack a matching verified completion record or their inputs changed; use a new run")
    return None


def validate_frozen_selection(config):
    """Gate test generation using development evidence; no generated test reads."""
    c = config_dict(config)
    reference_path, manifest, archive, provenance = _reference_context(c)
    selection = _selection_guard(c, manifest["manifest_hash"], stable_hash(provenance))
    groups, generated_provenance = load_generated_features(c, "development")
    if generated_provenance != provenance:
        raise ValueError("Frozen development/reference feature pipelines differ")
    fingerprint = _evaluation_fingerprint(c, "development", reference_path, manifest["manifest_hash"], provenance)
    _completed_metrics(c, "development", fingerprint, frozen=True)
    return selection


def evaluate_run(config, split):
    c = config_dict(config)
    if split not in ("development", "test"):
        raise ValueError("Metrics split must be development or test")
    result_dir = Path(c["output_dir"]) / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        reference_path, manifest, archive, provenance = _reference_context(c)
    except FileNotFoundError as error:
        atomic_json(result_dir / f"{split}_evaluation_status.json", {"status": "blocked_missing_reference", "reason": str(error)})
        raise
    real = _features(archive[split], "reference features")
    manifest_hash = manifest["manifest_hash"]
    provenance_hash = stable_hash(provenance)
    if split == "test":
        validate_frozen_selection(c)
    groups, generated_provenance = load_generated_features(c, split)
    if generated_provenance != provenance:
        raise ValueError("Generated/reference feature preprocessing or weights differ")
    fingerprint = _evaluation_fingerprint(c, split, reference_path, manifest_hash, provenance)
    frozen = split == "development" and (result_dir / "selected_k.json").exists()
    if frozen:
        _selection_guard(c, manifest_hash, provenance_hash)
    completed = _completed_metrics(c, split, fingerprint, frozen=frozen)
    if completed:
        return completed
    settings = _metric_settings(c)
    reference_metrics = ReferenceMetrics(real, settings["neighbors"])
    schedules, rows = {}, []
    for (method, k, generation, km), entries in sorted(groups.items(), key=lambda item: str(item[0])):
        metadata, features = zip(*entries)
        generated = np.asarray(features, dtype=np.float64)
        count = len(generated)
        if count not in schedules:
            schedule = kid_subset_schedule(len(real), count, settings["subset_size"], settings["subsets"], settings["seed"])
            schedule_hash = stable_hash({"real": schedule[0].tolist(), "generated": schedule[1].tolist()})
            schedules[count] = schedule, schedule_hash
            # Exact schedules retained, rather than just an RNG seed.
            atomic_json(result_dir / f"{split}_kid_schedule_{count}.json", {
                "reference_indices": schedule[0].tolist(), "generated_indices": schedule[1].tolist(),
                "schedule_hash": schedule_hash, "pool_order": [list(_pool_key(row)) for row in metadata]})
        schedule, schedule_hash = schedules[count]
        started = time.perf_counter()
        kid, kid_std = reference_metrics.kid(generated, schedule)
        fid = reference_metrics.fid(generated)
        precision, recall = reference_metrics.precision_recall(generated)
        metric_seconds = time.perf_counter() - started
        row = dict.fromkeys(METRIC_FIELDS)
        row.update(run_id=c["run_id"], preset=c["preset"], split=split, method=method, k=k,
                   generation_seed=generation, kmeans_seed=km, generated_count=count, reference_count=len(real),
                   kid=kid, kid_subset_std=kid_std, fid=fid, generative_precision=precision, generative_recall=recall,
                   mean_selected_cluster_size=_mean(metadata, "selected_cluster_size"),
                   mean_selected_cluster_fraction=_mean(metadata, "selected_cluster_fraction"),
                   mean_cluster_dispersion=_mean(metadata, "selected_cluster_dispersion"),
                   mean_full_pool_dispersion=_mean(metadata, "full_pool_dispersion"),
                   winner_tie_frequency=_mean([{"tie": int(int(x["winner_tie_count"]) > 1)} for x in metadata if x.get("winner_tie_count") is not None], "tie"),
                   convergence_failure_count=sum(not bool(x["converged"]) for x in metadata if x.get("converged") is not None),
                   total_candidate_evaluations=count * int(c.get("candidates_per_pool", 64)),
                   generator_seconds=_sum(metadata, "generator_seconds"), decoder_seconds=_sum(metadata, "decoder_seconds"),
                   clustering_seconds=_sum(metadata, "clustering_seconds"),
                   **_runtime_fields(metadata, metric_seconds),
                   peak_gpu_memory_gib=max((float(x["peak_gpu_memory_gib"]) for x in metadata if x.get("peak_gpu_memory_gib") is not None), default=None),
                   config_hash=c["config_hash"], reference_manifest_hash=manifest_hash,
                   generated_pool_manifest_hash=stable_hash([list(_pool_key(x)) for x in metadata]),
                   feature_provenance_hash=provenance_hash, kid_schedule_hash=schedule_hash,
                   kid_subset_size=settings["subset_size"], kid_subsets=settings["subsets"], precision_recall_neighbors=settings["neighbors"])
        rows.append(row)
    path = result_dir / f"{split}_metrics.csv"
    if split == "development" and (result_dir / "selected_k.json").exists():
        raise RuntimeError("Development selection was frozen concurrently; do not overwrite its metrics")
    atomic_csv(path, rows, METRIC_FIELDS)
    atomic_json(result_dir / f"{split}_evaluation_status.json", {"status": "complete", "rows": len(rows), "metrics_csv": str(path), "metrics_csv_sha256": file_sha256(path), "input_fingerprint": fingerprint, "feature_provenance": provenance, "metric_settings": settings, "smoke_only": c.get("preset") == "smoke"})
    return path


def select_k(config, development_csv=None):
    """Freeze separate arm choices. Never read test metrics to make a decision."""
    c = config_dict(config)
    result_dir = Path(c["output_dir"]) / "results"
    path = Path(development_csv or result_dir / "development_metrics.csv")
    selection_path = result_dir / "selected_k.json"
    if c.get("preset") == "smoke":
        raise ValueError("Synthetic/tiny smoke runs cannot select a scientific winning K; use real development data under main or compact protocol")
    rows = read_csv(path)
    if not rows or any(row["split"] != "development" for row in rows):
        raise ValueError("K selection accepts development metrics only")
    if any(row["config_hash"] != c["config_hash"] for row in rows):
        raise ValueError("Development/config hash mismatch")
    hashes = {key: {row[key] for row in rows} for key in ("reference_manifest_hash", "feature_provenance_hash")}
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Development rows mix reference manifests or feature pipelines")
    scores, choices = {}, {}
    for method in MAIN_METHODS:
        by_k_generation = defaultdict(list)
        seen = set()
        method_rows = [row for row in rows if row["method"] == method]
        for row in method_rows:
            k, seed, km = int(row["k"]), int(row["generation_seed"]), row["kmeans_seed"]
            identity = k, seed, km
            if identity in seen or (k == 1 and km != ""):
                raise ValueError("Duplicated development rows or clustering-seed repetitions at K=1")
            seen.add(identity)
            value = float(row["kid"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite development KID")
            by_k_generation[k, seed].append(value)
        expected_k = set(map(int, c.get("k_values", sorted({key[0] for key in by_k_generation}))))
        expected_generation = set(map(int, c.get("generation_seeds", sorted({key[1] for key in by_k_generation}))))
        expected_km = set(map(int, c.get("kmeans_seeds", [11, 29, 47])))
        if not expected_k or 1 not in expected_k:
            raise ValueError("Development selection requires a declared sweep including K=1")
        for k in expected_k:
            for generation in expected_generation:
                actual = {row["kmeans_seed"] for row in method_rows if int(row["k"]) == k and int(row["generation_seed"]) == generation}
                expected = {""} if k == 1 else {str(x) for x in expected_km}
                if actual != expected:
                    raise ValueError(f"Incomplete development sweep: {method}, K={k}, generation={generation}; got clustering seeds {actual}, expected {expected}")
        if {key[0] for key in by_k_generation} != expected_k or {key[1] for key in by_k_generation} != expected_generation:
            raise ValueError("Unexpected development K/generation settings")
        per_k = {}
        for k in sorted(expected_k):
            by_generation = {str(seed): float(np.mean(by_k_generation[k, seed])) for seed in sorted(expected_generation)}
            per_k[str(k)] = {"mean_development_kid": float(np.mean(list(by_generation.values()))), "generation_seed_means": by_generation}
        minimum = min(value["mean_development_kid"] for value in per_k.values())
        tolerance = 1e-12
        choices[method] = min(int(k) for k, value in per_k.items() if value["mean_development_kid"] <= minimum + tolerance)
        scores[method] = per_k
    payload = {"schema_version": 1, "run_id": c["run_id"], "preset": c["preset"], "config_hash": c["config_hash"],
               "reference_manifest_hash": next(iter(hashes["reference_manifest_hash"])),
               "feature_provenance_hash": next(iter(hashes["feature_provenance_hash"])),
               "metric": "unbiased polynomial KID: ((a dot b)/d+1)^3; self diagonals excluded; raw, unclamped",
               "metric_settings": _metric_settings(c),
               "decision_rule": "average clustering seeds within each generation seed, then average generation seeds; minimize development KID; smaller K for raw difference <= 1e-12",
               "tie_tolerance": 1e-12, "selected_k": choices, "scores": scores,
               "development_metrics_sha256": file_sha256(path), "development_metrics_path": str(path.resolve()),
               "test_inspected_for_selection": False}
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != payload:
            raise RuntimeError("Frozen selection would change; use a new run directory")
        return selection_path
    if (result_dir / "test_metrics.csv").exists():
        raise RuntimeError("Cannot create a new development selection after test metrics exist")
    # Atomic hard-link promotion is exclusive and never exposes partial JSON.
    # It also prevents simultaneous selectors from replacing a frozen decision.
    result_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="selected_k.", suffix=".tmp", dir=result_dir)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, selection_path)
        except FileExistsError:
            if json.loads(selection_path.read_text()) != payload:
                raise RuntimeError("Concurrent selection differs from the frozen decision")
    finally:
        os.unlink(temporary)
    return selection_path
