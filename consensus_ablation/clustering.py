"""Ordinary Euclidean K-means over complete candidate tensors.

Lloyd assignments and centroid distances are computed from the exact full-
coordinate Gram matrix, accumulated in float64. A common coordinate-wise pool
translation improves stability and preserves all Euclidean distances. There is
no feature extraction, normalization, spatial reduction, or PCA here.

Empty clusters are repaired by moving the highest-residual eligible point
whose donor has at least two members; ties use the original candidate index.
On identical samples this can create deterministic, nonempty zero-dispersion
groups; the number of numerically distinct samples and degeneracy are recorded.
Convergence requires unchanged repaired assignments, never a centroid tolerance.
"""

import time

import numpy as np


WINNER_TIE_ATOL = 1e-10
WINNER_TIE_RTOL = 1e-10


def prepare_gram(samples, *, coordinate_chunk_size=8192):
    """Compute XX^T using every coordinate, after one common pool translation."""
    samples = np.asarray(samples)
    if samples.ndim < 2 or samples.shape[0] < 1:
        raise ValueError("Expected shape [candidate, ...complete tensor coordinates]")
    if coordinate_chunk_size < 1:
        raise ValueError("coordinate_chunk_size must be positive")
    flat = samples.reshape(samples.shape[0], -1)
    if not flat.shape[1]:
        raise ValueError("Candidate tensors must contain coordinates")
    gram = np.zeros((len(flat), len(flat)), dtype=np.float64)
    for start in range(0, flat.shape[1], coordinate_chunk_size):
        block = flat[:, start:start + coordinate_chunk_size].astype(np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError("Candidate tensors contain nonfinite coordinates")
        block -= block.mean(axis=0, keepdims=True)
        gram += block @ block.T
    return (gram + gram.T) * 0.5


def _validate_gram(gram):
    original = np.asarray(gram)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram):
        raise ValueError("Gram matrix must be nonempty and square")
    if not np.all(np.isfinite(gram)):
        raise ValueError("Gram matrix contains nonfinite values")
    precision = (np.finfo(original.dtype).eps
                 if np.issubdtype(original.dtype, np.floating) else np.finfo(np.float64).eps)
    scale = max(1.0, float(np.max(np.abs(gram))))
    tolerance = 128 * precision * scale
    if not np.allclose(gram, gram.T, atol=tolerance, rtol=0):
        raise ValueError("Gram matrix must be symmetric")
    return (gram + gram.T) * 0.5, tolerance


def _clamp_roundoff(values, tolerance):
    if float(np.min(values)) < -tolerance:
        raise ValueError("Significant negative squared distance; Gram arithmetic is invalid")
    return np.maximum(values, 0.0)


def _point_distances(gram, tolerance):
    diagonal = np.diag(gram)
    return _clamp_roundoff(diagonal[:, None] + diagonal[None, :] - 2 * gram,
                          tolerance)


def _centroid_distances(gram, labels, k, tolerance):
    membership = (labels[:, None] == np.arange(k)[None, :]).astype(np.float64)
    counts = membership.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Repair empty clusters before computing centroids")
    products = gram @ membership
    centroid_norms = np.sum(membership * products, axis=0) / counts ** 2
    distances = (np.diag(gram)[:, None] - 2 * products / counts
                 + centroid_norms[None, :])
    return _clamp_roundoff(distances, tolerance)


def gram_centroid_distances(gram, labels, k=None):
    """Public audit helper: full-coordinate squared distances to label means."""
    gram, tolerance = _validate_gram(gram)
    labels = np.asarray(labels)
    if labels.shape != (len(gram),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be one integer per candidate")
    k = int(np.max(labels)) + 1 if k is None else int(k)
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Labels are outside the requested cluster range")
    return _centroid_distances(gram, labels, k, tolerance)


def _ids(candidate_ids, count):
    ids = ([f"{index:012d}" for index in range(count)]
           if candidate_ids is None else list(candidate_ids))
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("Candidate IDs must be unique and match the pool size")
    if not all(isinstance(value, str) for value in ids):
        raise ValueError("Candidate IDs must be strings")
    return ids


def select_largest_cluster(labels, *, gram=None, squared_distances=None,
                           candidate_ids=None, k=None,
                           tie_atol=WINNER_TIE_ATOL, tie_rtol=WINNER_TIE_RTOL):
    """Cardinality, mean squared dispersion, then sorted candidate-ID tuple.

    ``winner_tie_count`` counts cardinality ties, before dispersion and IDs.
    The final comparison is independent of arbitrary numeric cluster labels.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a nonempty integer vector")
    k = int(labels.max()) + 1 if k is None else int(k)
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Cluster label is outside [0,K)")
    ids = _ids(candidate_ids, len(labels))
    counts = np.bincount(labels, minlength=k)
    if squared_distances is None:
        if gram is None:
            raise ValueError("Gram or squared centroid distances are required for ties")
        squared_distances = gram_centroid_distances(gram, labels, k)
    distances = np.asarray(squared_distances, dtype=np.float64)
    if distances.shape != (len(labels), k) or not np.all(np.isfinite(distances)):
        raise ValueError("Expected finite [candidate,K] squared centroid distances")
    dispersions = np.full(k, np.inf)
    inertias = np.zeros(k, dtype=np.float64)
    for cluster in range(k):
        indices = np.flatnonzero(labels == cluster)
        if len(indices):
            inertias[cluster] = distances[indices, cluster].sum(dtype=np.float64)
            dispersions[cluster] = inertias[cluster] / len(indices)
    cardinality_ties = np.flatnonzero(counts == counts.max())
    minimum = min(dispersions[cardinality_ties])
    finalists = [int(cluster) for cluster in cardinality_ties
                 if np.isclose(dispersions[cluster], minimum, rtol=tie_rtol, atol=tie_atol)]
    winner = min(finalists, key=lambda cluster: tuple(sorted(
        ids[index] for index in np.flatnonzero(labels == cluster))))
    selected = np.flatnonzero(labels == winner)
    return {
        "selected_cluster_label": int(winner),
        "selected_indices": selected.tolist(),
        "selected_candidate_ids": [ids[index] for index in selected],
        "selected_cluster_size": int(len(selected)),
        "selected_cluster_fraction": float(len(selected) / len(labels)),
        "cluster_sizes": counts.tolist(),
        "cluster_inertias": inertias.tolist(),
        "cluster_mean_squared_distances": [float(x) if np.isfinite(x) else None
                                            for x in dispersions],
        "selected_cluster_dispersion": float(dispersions[winner]),
        "winner_tie_count": int(len(cardinality_ties)),
        "dispersion_tie_count": len(finalists),
        "winner_tie_atol": float(tie_atol),
        "winner_tie_rtol": float(tie_rtol),
    }


def _initialize(point_distances, k, rng):
    count = len(point_distances)
    centers = [int(rng.integers(count))]
    closest = point_distances[:, centers[0]].copy()
    for _ in range(1, k):
        closest[centers] = 0.0
        total = float(closest.sum(dtype=np.float64))
        if total > 0:
            index = int(rng.choice(count, p=closest / total))
        else:
            index = next(index for index in range(count) if index not in centers)
        centers.append(index)
        closest = np.minimum(closest, point_distances[:, index])
    return centers


def _repair_empty(labels, distances, k):
    labels = labels.copy()
    counts = np.bincount(labels, minlength=k)
    residuals = distances[np.arange(len(labels)), labels]
    repairs = 0
    for empty in np.flatnonzero(counts == 0):
        eligible = np.flatnonzero(counts[labels] > 1)
        if not len(eligible):
            raise RuntimeError("Cannot repair empty cluster without reducing K")
        # np.argmax returns the first candidate index for exactly tied residuals.
        candidate = int(eligible[np.argmax(residuals[eligible])])
        donor = labels[candidate]
        labels[candidate] = empty
        counts[donor] -= 1
        counts[empty] += 1
        repairs += 1
    return labels, repairs


def _canonical_labels(labels, ids, k):
    clusters = sorted(range(k), key=lambda cluster: tuple(sorted(
        ids[index] for index in np.flatnonzero(labels == cluster))))
    remap = np.empty(k, dtype=np.int64)
    for new, old in enumerate(clusters):
        remap[old] = new
    return remap[labels]


def fit_kmeans_from_gram(gram, k, seed, *, n_init=10, max_iter=100,
                         candidate_ids=None, initial_center_indices=None):
    """Fit full Lloyd K-means with K-means++ and minimum-inertia restarts.

    ``initial_center_indices`` is an audit hook: one list for one restart, or
    one list per requested restart. It permits comparison with direct Euclidean
    Lloyd updates with precisely the same initial centers.
    """
    started = time.perf_counter()
    gram, tolerance = _validate_gram(gram)
    count = len(gram)
    if isinstance(k, bool) or int(k) != k or not 1 <= k <= count:
        raise ValueError("K must be an integer in [1,candidate_count]")
    k = int(k)
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    ids = _ids(candidate_ids, count)
    point_distances = _point_distances(gram, tolerance)
    representatives = []
    for index in range(count):
        if not any(point_distances[index, rep] <= tolerance for rep in representatives):
            representatives.append(index)
    common = {
        "candidate_count": count, "k": k, "kmeans_seed": int(seed) if k > 1 else None,
        "n_init": int(n_init) if k > 1 else 0, "max_iter": int(max_iter),
        "algorithm": "full_coordinate_euclidean_lloyd_kmeans++",
        "arithmetic": "float64_gram_and_centroid_reductions",
        "convergence_rule": "unchanged_repaired_assignments",
        "empty_cluster_repair": "highest_residual_eligible_point_then_candidate_index",
        "distinct_sample_count": len(representatives),
        "distinctness_squared_distance_tolerance": float(tolerance),
        "degenerate_fewer_distinct_than_k": len(representatives) < k,
    }
    if k == 1:
        labels = np.zeros(count, dtype=np.int64)
        distances = _centroid_distances(gram, labels, k, tolerance)
        result = {**common, "labels": labels.tolist(), "iterations": 0,
                  "converged": True, "iteration_limit_reached": False,
                  "empty_cluster_reinitializations": 0, "chosen_restart": None,
                  "restart_inertias": [], "initial_center_indices": [],
                  "inertia": float(distances.sum(dtype=np.float64)),
                  **select_largest_cluster(labels, squared_distances=distances,
                                            candidate_ids=ids, k=k)}
        result["elapsed_seconds"] = time.perf_counter() - started
        return result
    provided = None
    if initial_center_indices is not None:
        provided = np.asarray(initial_center_indices)
        if provided.ndim == 1:
            provided = provided[None, :]
        if provided.shape != (n_init, k) or not np.issubdtype(provided.dtype, np.integer):
            raise ValueError("Initial centers must have shape [n_init,K]")
        if np.any(provided < 0) or np.any(provided >= count):
            raise ValueError("Initial center index outside the candidate pool")
        if any(len(set(row.tolist())) != k for row in provided):
            raise ValueError("Initial center indices must be distinct")
    restart_seeds = np.random.SeedSequence(int(seed)).spawn(n_init)
    best = None
    restart_inertias = []
    for restart, restart_seed in enumerate(restart_seeds):
        centers = (provided[restart].tolist() if provided is not None else
                   _initialize(point_distances, k, np.random.default_rng(restart_seed)))
        distances = point_distances[:, centers]
        previous = None
        converged = False
        repairs = 0
        for iteration in range(1, max_iter + 1):
            labels, repaired = _repair_empty(np.argmin(distances, axis=1), distances, k)
            repairs += repaired
            distances = _centroid_distances(gram, labels, k, tolerance)
            if previous is not None and np.array_equal(labels, previous):
                converged = True
                break
            previous = labels.copy()
        labels = _canonical_labels(labels, ids, k)
        distances = _centroid_distances(gram, labels, k, tolerance)
        inertia = float(distances[np.arange(count), labels].sum(dtype=np.float64))
        restart_inertias.append(inertia)
        partition_key = tuple(tuple(sorted(ids[index] for index in np.flatnonzero(labels == cluster)))
                              for cluster in range(k))
        proposal = {"labels": labels.tolist(), "inertia": inertia,
                    "iterations": iteration, "converged": converged,
                    "iteration_limit_reached": not converged,
                    "empty_cluster_reinitializations": repairs,
                    "chosen_restart": restart, "initial_center_indices": centers,
                    **select_largest_cluster(labels, squared_distances=distances,
                                              candidate_ids=ids, k=k)}
        if best is None or (inertia, partition_key) < best[0]:
            best = ((inertia, partition_key), proposal)
    result = {**common, **best[1], "restart_inertias": restart_inertias}
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def fit_kmeans(samples, k, seed, *, n_init=10, max_iter=100,
                candidate_ids=None, gram=None, initial_center_indices=None):
    samples = np.asarray(samples)
    if samples.ndim < 2:
        raise ValueError("Expected complete candidate tensors with a leading candidate axis")
    if gram is None:
        gram = prepare_gram(samples)
    elif np.asarray(gram).shape != (len(samples), len(samples)):
        raise ValueError("Cached Gram matrix does not match candidate count")
    return fit_kmeans_from_gram(gram, k, seed, n_init=n_init, max_iter=max_iter,
                                candidate_ids=candidate_ids,
                                initial_center_indices=initial_center_indices)
