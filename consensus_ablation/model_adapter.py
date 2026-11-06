"""Inference-only adapter around the pinned official JAX MeanFlow implementation."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shlex

import numpy as np

from .assets import load_assets


NUMERICS_DEFAULTS = {
    "matmul_precision": "highest",
    "xla_gpu_autotune_level": 4,
    "xla_gpu_deterministic_ops": False,
    "xla_gpu_exclude_nondeterministic_ops": False,
    "persistent_compilation_cache": True,
}

JAX_CACHE_SETTINGS = {
    "jax_persistent_cache_min_compile_time_secs": 0.0,
    "jax_persistent_cache_min_entry_size_bytes": -1,
    "jax_persistent_cache_enable_xla_caches": "xla_gpu_per_fusion_autotune_cache_dir",
    "jax_raise_persistent_cache_errors": True,
}


def configure_numerics_environment(config, *, environ=None):
    """Resolve the fixed FP32 policy before JAX parses backend flags.

    This helper intentionally imports no JAX code, so configuration conflicts
    fail before a CUDA context or an expensive model restoration is created.
    Preserve unrelated XLA arguments, including quoted argument values.
    """
    environ = os.environ if environ is None else environ
    supplied = config.get("numerics", {})
    if not isinstance(supplied, dict):
        raise ValueError("numerics must be a mapping defining the FP32 numerical policy")
    unknown = set(supplied) - set(NUMERICS_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown numerical policy fields: {sorted(unknown)}")
    policy = {**NUMERICS_DEFAULTS, **supplied}
    for name, expected in NUMERICS_DEFAULTS.items():
        value = policy[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"This protocol requires numerics.{name}={expected!r}; received {value!r}")
    precision = environ.get("JAX_DEFAULT_MATMUL_PRECISION")
    if precision is not None and precision not in ("highest", "float32"):
        raise ValueError("JAX_DEFAULT_MATMUL_PRECISION conflicts with the required highest/FP32 policy")
    try:
        arguments = shlex.split(environ.get("XLA_FLAGS", ""))
    except ValueError as error:
        raise ValueError("XLA_FLAGS contains invalid shell quoting") from error
    required = {name: value for name, value in policy.items() if name.startswith("xla_")}
    preserved = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        key, separator, value = argument.partition("=")
        name = key.removeprefix("--") if key.startswith("--") else ""
        negated = name.startswith("no") and name[2:] in required
        if negated:
            name = name[2:]
            if not isinstance(required[name], bool) or separator:
                raise ValueError(f"XLA_FLAGS {key} is not a valid numerical policy override")
            value, separator = "false", "="
        if name not in required:
            preserved.append(argument)
            index += 1
            continue
        expected = required[name]
        if not separator:
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                index += 1
                value = arguments[index]
            elif isinstance(expected, bool):
                value = "true"
            else:
                raise ValueError(f"XLA_FLAGS {key} requires an explicit value")
        accepted = ({"true", "1"} if expected else {"false", "0"}) if isinstance(expected, bool) else {str(expected)}
        if value.lower() not in accepted:
            raise ValueError(f"XLA_FLAGS {key}={value} conflicts with required value {expected!r}")
        index += 1
    for name, value in required.items():
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        preserved.append(f"--{name}={encoded}")
    effective_flags = shlex.join(preserved)
    paths = config.get("paths", {})
    assets_root = config.get("assets_root") or paths.get("assets_root", "work/meanflow-assets")
    cache_dir = str(Path(paths.get("compilation_cache") or
        Path(assets_root) / "jax-cache-fp32-v3-jax0.8-B200").expanduser().resolve())
    updates = {"JAX_DEFAULT_MATMUL_PRECISION": policy["matmul_precision"],
        "XLA_FLAGS": effective_flags, "JAX_COMPILATION_CACHE_DIR": cache_dir,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    for name, value in JAX_CACHE_SETTINGS.items():
        updates[name.upper()] = str(value).lower() if isinstance(value, bool) else str(value)
    # A persistent executable and its per-fusion choices define this policy.
    # Reject outside overrides rather than silently running without the cache.
    for name, expected in updates.items():
        if name in ("JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS"):
            continue
        value = environ.get(name)
        if value is None:
            continue
        if name == "JAX_COMPILATION_CACHE_DIR":
            equal = str(Path(value).expanduser().resolve()) == expected
        elif name == "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS":
            try:
                equal = float(value) == 0.0
            except ValueError:
                equal = False
        elif name == "JAX_RAISE_PERSISTENT_CACHE_ERRORS":
            equal = value.lower() in ("true", "1")
        else:
            equal = value == expected
        if not equal:
            raise ValueError(f"{name} conflicts with the required persistent FP32 numerical policy")
    # Commit environment changes only after every conflict has been checked.
    environ.update(updates)
    return {"policy": policy, "xla_flags": effective_flags,
        "compilation_cache_dir": cache_dir, "jax_cache_settings": dict(JAX_CACHE_SETTINGS),
        "allocator": updates["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "cublas_workspace_config": updates["CUBLAS_WORKSPACE_CONFIG"]}


def _float32_weight(value, label):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise RuntimeError(f"{label} must contain original float32 weights; received {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite {label}")
    return array


class MeanFlowAdapter:
    """FP32, EMA, one-step inference and exact raw floating VAE decoding."""

    def __init__(self, config):
        numerical_environment = configure_numerics_environment(config)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        from jax._src import xla_bridge
        if xla_bridge.backends_are_initialized():
            raise RuntimeError("Initialize MeanFlowAdapter in a fresh process before any JAX backend; numerical flags must be parsed before device initialization")
        jax.config.update("jax_default_matmul_precision", numerical_environment["policy"]["matmul_precision"])
        Path(numerical_environment["compilation_cache_dir"]).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", numerical_environment["compilation_cache_dir"])
        for name, value in numerical_environment["jax_cache_settings"].items():
            jax.config.update(name, value)
        import jax.numpy as jnp
        from flax.training import checkpoints
        from flax.traverse_util import flatten_dict
        from diffusers import FlaxAutoencoderKL
        from meanflow import MeanFlow

        self.config = config
        self.assets = load_assets(config)
        self.jax, self.jnp = jax, jnp
        self.microbatch_size = int(config.get("microbatch_size", 16))
        self.decode_batch_size = int(config.get("decode_batch_size", self.microbatch_size))
        if self.microbatch_size < 1 or self.decode_batch_size < 1:
            raise ValueError("microbatch_size and decode_batch_size must be positive")
        devices = jax.local_devices()
        if not config.get("allow_cpu", False) and not any(item.platform == "gpu" for item in devices):
            raise RuntimeError("No JAX GPU is available in this authorized allocation")
        if sum(item.platform == "gpu" for item in devices) > 1:
            raise RuntimeError("This campaign is limited to one visible GPU")
        self.validation = {"inference_only": True, "ema": True, "native_layout": "NHWC",
            "decoder_input_layout": "NCHW", "decode_output_layout": "NHWC",
            "t": 1.0, "h": 1.0, "sampling_equivalence_checked": False}
        self.model = MeanFlow(model_str="DiT_B_4", model_config={}, num_steps=1,
            schedule="default", guidance_eq="cfg", omega=1.0, kappa=0.5)
        restored = checkpoints.restore_checkpoint(self.assets["checkpoint_dir"], target=None)
        if not isinstance(restored, dict) or "ema_params" not in restored:
            raise RuntimeError(f"Official checkpoint did not restore an EMA parameter tree; restored keys: {list(restored) if isinstance(restored, dict) else type(restored).__name__}")
        params = restored["ema_params"]
        step = int(np.asarray(restored.get("step", -1)))
        if step < 1:
            raise RuntimeError("Official checkpoint restoration did not recover a trained positive step")
        expected = jax.eval_shape(lambda: self.model.init({"params": jax.random.PRNGKey(0)},
            jnp.ones((1, 32, 32, 4), jnp.float32), jnp.ones((1,), jnp.int32),
            jnp.ones((1,), jnp.int32)))["params"]
        flat_expected, flat_params = flatten_dict(expected), flatten_dict(params)
        if set(flat_expected) != set(flat_params):
            missing = sorted(set(flat_expected) - set(flat_params))[:5]
            extra = sorted(set(flat_params) - set(flat_expected))[:5]
            raise RuntimeError(f"EMA parameter keys do not match official DiT_B_4; missing={missing}, extra={extra}")
        for key, value in flat_params.items():
            if tuple(np.shape(value)) != tuple(flat_expected[key].shape):
                raise RuntimeError(f"EMA parameter shape mismatch at {'/'.join(key)}")
            _float32_weight(value, f"EMA parameter at {'/'.join(key)}")
        self.params = jax.device_put(jax.tree_util.tree_map(np.asarray, params))
        del restored
        self.validation["parameter_shapes_validated"] = len(flat_params)
        self.validation["checkpoint_step"] = step

        def predict(epsilon, labels):
            t = jnp.ones((epsilon.shape[0],), dtype=jnp.float32)
            return self.model.apply({"params": self.params}, epsilon, t, t, labels,
                train=False, method=self.model.u_fn, rngs={"gen": jax.random.PRNGKey(0)})

        def official_endpoint(epsilon, labels):
            t = jnp.ones((epsilon.shape[0],), dtype=jnp.float32)
            return self.model.apply({"params": self.params}, epsilon, t, jnp.zeros_like(t), labels,
                method=self.model.solver_step, rngs={"gen": jax.random.PRNGKey(0)})

        self._predict = jax.jit(predict)
        self._official_endpoint = jax.jit(official_endpoint)
        self.vae, self.vae_params = FlaxAutoencoderKL.from_pretrained(
            self.assets["vae_path"], dtype=jnp.float32, local_files_only=True)
        self.vae_params = jax.device_put(jax.tree_util.tree_map(lambda value: _float32_weight(value, "VAE parameters"), self.vae_params))

        def decode_raw(latents):
            nchw = jnp.transpose(latents, (0, 3, 1, 2)) / jnp.asarray(0.18215, jnp.float32)
            decoded = self.vae.apply({"params": self.vae_params}, nchw,
                method=FlaxAutoencoderKL.decode).sample
            return jnp.transpose(decoded, (0, 2, 3, 1))

        self._decode = jax.jit(decode_raw)
        self.provenance = {**self.assets, "architecture": "DiT_B_4", "checkpoint_step": step,
            "compute_dtype": "float32", "accumulation_dtype": "float32", "ema": True,
            "native_guidance": "single official u_fn pass; cfg omega=1,kappa=0.5 training configuration",
            "devices": [str(item) for item in devices], "microbatch_size": self.microbatch_size,
            "decode_batch_size": self.decode_batch_size,
            "numerical_policy": numerical_environment["policy"],
            "xla_flags": numerical_environment["xla_flags"],
            "compilation_cache_dir": numerical_environment["compilation_cache_dir"],
            "jax_cache_settings": numerical_environment["jax_cache_settings"],
            "cublas_workspace_config": numerical_environment["cublas_workspace_config"],
            "jax_allocator": numerical_environment["allocator"],
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "diffusers")}}

    def generate(self, epsilon, class_id):
        epsilon = self._check_latents(epsilon)
        if not 0 <= int(class_id) < 1000:
            raise ValueError("class_id must be an ImageNet class index in [0,1000)")
        velocities, endpoints = [], []
        for begin in range(0, len(epsilon), self.microbatch_size):
            current = epsilon[begin:begin + self.microbatch_size]
            count = len(current)
            padded = np.pad(current, ((0, self.microbatch_size - count), (0, 0), (0, 0), (0, 0)))
            labels = np.full(self.microbatch_size, int(class_id), dtype=np.int32)
            velocity = np.asarray(self.jax.device_get(self._predict(padded, labels)), dtype=np.float32)
            endpoint = padded - velocity
            if not self.validation["sampling_equivalence_checked"]:
                official = np.asarray(self.jax.device_get(self._official_endpoint(padded, labels)), dtype=np.float32)
                maximum = float(np.max(np.abs(endpoint - official)))
                np.testing.assert_allclose(endpoint, official, rtol=2e-5, atol=2e-5,
                    err_msg="Captured u_fn does not reproduce the official one-step solver")
                self.validation.update(sampling_equivalence_checked=True,
                    sampling_equivalence_max_abs_error=maximum,
                    sampling_equivalence_rtol=2e-5, sampling_equivalence_atol=2e-5)
            if not np.isfinite(velocity[:count]).all() or not np.isfinite(endpoint[:count]).all():
                raise RuntimeError("Non-finite MeanFlow output")
            velocities.append(velocity[:count])
            endpoints.append(endpoint[:count])
        return np.concatenate(velocities), np.concatenate(endpoints)

    def decode(self, latents):
        latents = self._check_latents(latents)
        outputs = []
        for begin in range(0, len(latents), self.decode_batch_size):
            current = latents[begin:begin + self.decode_batch_size]
            count = len(current)
            padded = np.pad(current, ((0, self.decode_batch_size - count), (0, 0), (0, 0), (0, 0)))
            decoded = np.asarray(self.jax.device_get(self._decode(padded)), dtype=np.float32)[:count]
            if decoded.shape != (count, 256, 256, 3) or not np.isfinite(decoded).all():
                raise RuntimeError(f"Invalid raw VAE image shape or values: {decoded.shape}")
            outputs.append(decoded)
        return np.concatenate(outputs)

    @staticmethod
    def _check_latents(value):
        result = np.asarray(value, dtype=np.float32)
        if result.ndim != 4 or result.shape[1:] != (32, 32, 4) or len(result) == 0:
            raise ValueError(f"Expected nonempty (N,32,32,4) native latents, received {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("Input latents contain non-finite values")
        return result
