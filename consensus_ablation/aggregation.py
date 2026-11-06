"""Raw floating consensus reductions, with explicit noise/velocity pairing."""

import numpy as np


def _selected(values, indices):
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[0] < 1:
        raise ValueError("Expected a nonempty candidate axis and tensor coordinates")
    raw_indices = np.asarray(indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise ValueError("Selected candidate indices must be a nonempty vector")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("Selected candidate indices must be integers")
    if np.any(raw_indices < 0) or np.any(raw_indices >= values.shape[0]):
        raise ValueError("Selected candidate index is outside the pool")
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise ValueError("Selected candidate indices contain duplicates")
    chosen = values[raw_indices]
    if not np.all(np.isfinite(chosen)):
        raise ValueError("Selected tensors contain nonfinite coordinates")
    return chosen


def image_mean(images, indices):
    """Mean the complete raw decoder tensors before clipping or quantization."""
    selected = _selected(images, indices)
    dtype = np.result_type(selected.dtype, np.float32)
    return selected.mean(axis=0, dtype=np.float64).astype(dtype)


def paired_latent_mean(epsilon, velocities, indices, *, epsilon_ids=None,
                       velocity_ids=None, endpoints=None):
    """Return mean(epsilon[C]) - mean(u[C]) using matching candidate IDs.

    Audit reductions use float64, even when the forward pass used float32.
    Endpoint tensors, if supplied, are verified against epsilon - u.
    """
    epsilon, velocities = np.asarray(epsilon), np.asarray(velocities)
    if epsilon.shape != velocities.shape:
        raise ValueError("Noise and velocity tensors must have identical shapes")
    if (epsilon_ids is None) != (velocity_ids is None):
        raise ValueError("Both noise and velocity candidate IDs must be supplied")
    if epsilon_ids is not None:
        if len(epsilon_ids) != len(epsilon) or len(velocity_ids) != len(velocities):
            raise ValueError("Candidate-ID count does not match tensor count")
        if list(epsilon_ids) != list(velocity_ids):
            raise ValueError("Noise and velocity candidate IDs are mismatched")
        if len(set(epsilon_ids)) != len(epsilon_ids):
            raise ValueError("Candidate IDs must be unique within a pool")
    selected_epsilon = _selected(epsilon, indices)
    selected_velocity = _selected(velocities, indices)
    if endpoints is not None:
        endpoints = np.asarray(endpoints)
        if endpoints.shape != epsilon.shape or not np.allclose(
                _selected(endpoints, indices),
                selected_epsilon.astype(np.float64) - selected_velocity.astype(np.float64),
                rtol=2e-6, atol=2e-6):
            raise ValueError("Endpoint tensors do not match the paired noise - velocity")
    latent = (selected_epsilon.mean(axis=0, dtype=np.float64)
              - selected_velocity.mean(axis=0, dtype=np.float64))
    return latent.astype(np.result_type(epsilon.dtype, velocities.dtype, np.float32))


def velocity_cluster_image_mean(images, indices):
    """Diagnostic: image-space mean with the saved velocity memberships."""
    return image_mean(images, indices)


def to_uint8(raw_images):
    """The audited upstream conversion, applied once to each final output."""
    raw_images = np.asarray(raw_images)
    if not np.all(np.isfinite(raw_images)):
        raise ValueError("Cannot export nonfinite decoder output")
    return np.clip(127.5 * raw_images + 128.0, 0.0, 255.0).astype(np.uint8)
