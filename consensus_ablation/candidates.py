"""Stable, batching-independent identities for model-space Gaussian draws.

The versioned canonical JSON tuple is hashed with SHA-256. The first eight
digest bytes, interpreted as two big-endian uint32 words, are the legacy JAX
PRNG key data. Pass ``np.asarray(candidate_key(identity), dtype=np.uint32)``
directly to ``jax.random.normal``; never redraw a batch from a pool-level key.
No JAX import is required to inspect or test candidate manifests.
"""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class CandidateIdentity:
    generation_seed: int
    split: str
    class_id: int
    group_id: int
    candidate_index: int
    purpose: str = "latent_noise"

    def __post_init__(self):
        for name in ("generation_seed", "class_id", "group_id", "candidate_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (not isinstance(self.split, str) or not self.split
                or not isinstance(self.purpose, str) or not self.purpose):
            raise ValueError("split and purpose must be nonempty strings")

    def canonical_bytes(self):
        return json.dumps(
            ["meanflow-candidate-v1", int(self.generation_seed), self.split,
             int(self.class_id), int(self.group_id), int(self.candidate_index), self.purpose],
            ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")


def candidate_id(identity: CandidateIdentity) -> str:
    return hashlib.sha256(identity.canonical_bytes()).hexdigest()


def candidate_key(identity: CandidateIdentity) -> tuple[int, int]:
    digest = hashlib.sha256(identity.canonical_bytes()).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big")


def pool_candidate_ids(split, generation_seed, class_id, group_id, count,
                       purpose="latent_noise"):
    if count < 1:
        raise ValueError("A candidate pool must contain at least one candidate")
    return [candidate_id(CandidateIdentity(generation_seed, split, class_id,
                                         group_id, index, purpose))
            for index in range(count)]


def pool_candidate_keys(split, generation_seed, class_id, group_id, count,
                        purpose="latent_noise"):
    if count < 1:
        raise ValueError("A candidate pool must contain at least one candidate")
    return [candidate_key(CandidateIdentity(generation_seed, split, class_id,
                                          group_id, index, purpose))
            for index in range(count)]
