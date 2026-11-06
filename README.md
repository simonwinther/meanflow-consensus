# MeanFlow: Average Image and Average Velocity

Two inference experiments using the released MeanFlow MF-B/4 EMA checkpoint on
ImageNet at 256 × 256 resolution:

- **Average Image (A):** cluster complete floating decoded images and average the largest cluster.
- **Average Velocity (B):** cluster complete predicted average velocities, average their matching noise–velocity pairs, and decode the mean endpoint latent.

For each candidate, `z = epsilon - u`. B computes
`D(mean(epsilon[C]) - mean(u[C]))`; A averages `D(z[C])` using image memberships.
A diagnostic averages images using B's memberships to isolate the nonlinear
decoder's effect. Reductions precede clipping and uint8 conversion. The ordinary
sampling control always uses candidate 0. Neither approach trains new parameters.

## Installation

Use Linux, [uv](https://docs.astral.sh/uv/) and Python 3.12. The completed run used
Python 3.12.3; installation and CPU checks also passed on 3.12.13. The inference
environment uses JAX/jaxlib 0.8.0 with CUDA 13, Flax 0.12.0 and Diffusers 0.35.1.
A separate feature environment uses PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128
and torch-fidelity 0.3.0. Both installed environments are pinned in
`scripts/consensus/environment-*.lock.txt`.

From the repository root:

```bash
bash scripts/consensus/setup.sh
source work/meanflow-env/bin/activate
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
python -m consensus_ablation inspect --config configs/consensus_main.yaml
```

The CA-bundle path above was tested on Ubuntu/NixOS; adjust it if your system
uses another location. Set `MEANFLOW_STORAGE` before installation and in later
shells to place environments, assets and runs outside the default `work/` root.
Generation requires one visible NVIDIA GPU with compatible drivers. Reports
from included CSVs require neither a GPU nor downloaded model/data assets.

## Checkpoint and ImageNet

```bash
python -m consensus_ablation prepare-assets --config configs/consensus_main.yaml
python -m consensus_ablation prepare-data --config configs/consensus_main.yaml
```

Asset preparation downloads the [official MF-B/4 checkpoint](https://drive.google.com/file/d/19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu/view)
(`b4_240ep_kappa0.5.zip`, 1,939,631,766 bytes; step 1,200,960) and the
[Flax VAE](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax/tree/7581b0a0489cc8483c876a728b830b9ce087cf03).
The checkpoint archive SHA-256 is checked against
`cfe35729ecf7dbf6aa4afc51933444a2a45886ce60a55453489e1e5c3b8ffd3c`;
the VAE revision is pinned to `7581b0a0489cc8483c876a728b830b9ce087cf03`.
Inference restores `ema_params`. Inception weights download on first extraction.
If Drive access fails, supply the authorized ZIP at
`work/meanflow-assets/b4_240ep_kappa0.5.zip` and rerun asset preparation.

Only ImageNet ILSVRC2012 **validation** data and the devkit are required. Obtain
access through [ImageNet](https://image-net.org/download.php) under its terms.
If the official download is unavailable, place the authorized archives here
and rerun data preparation:

```text
work/imagenet/ILSVRC2012_img_val.tar
work/imagenet/ILSVRC2012_devkit_t12.tar.gz
```

Preparation verifies the published MD5 values, translates labels through the
devkit, and extracts `work/imagenet/val/<synset>/ILSVRC2012_val_<index>.JPEG`.
The fixed 100 classes use `default_rng(1729)`, retaining 50 images each. Class-local
hashes of seed 2718 select 20 development and 30 test references per class.
References use RGB conversion and ADM-style resize/center crop to 256 × 256.
Training images and full-ImageNet FID statistics are not required.

No approach-specific retraining is needed. Training the released checkpoint
itself is outside this experiment; its [original training source](https://github.com/Gsunshine/meanflow/tree/d70cb55d298ee03c53bf6da67bec281082e4e2d9)
is available upstream. No locally verified checkpoint-retraining recipe is claimed.

## Run both approaches

After preparation, on an assigned GPU:

```bash
python -m consensus_ablation campaign --config configs/consensus_main.yaml
```

The campaign runs the numerical replay audit, smoke and benchmark stages,
reference extraction, development generation/scoring, separate K selection,
held-out generation/scoring, and reporting. It runs both arms and controls on
identical candidate pools. The benchmark chooses main if it fits the 48-hour
execution budget, otherwise the compact fallback. Compact is a different protocol;
check the saved decision. The launcher uses an assigned GPU and does not allocate one.

Main uses 100 classes, N=64 candidates per pool, K=[1,2,5,10,15], generation seeds
[2026,2027,2028], and clustering seeds [11,29,47]. There are 5 development and
20 test pools per class/seed: 480,000 candidate forwards total. Full-coordinate
Euclidean K-means uses ten K-means++ restarts and at most 100 Lloyd iterations,
retaining minimum inertia. Largest-cluster ties use dispersion, then candidate
IDs. K=1 uses all candidates with no clustering-seed repetitions. Compact uses
the first 20 classes and 2/10 development/test pools instead.

Outputs are under `work/meanflow-runs/<run_id>/`: candidate shards, memberships,
features, timings, audit images and `results/` tables/figures. A is
`image_cluster_image_mean`; B is `velocity_cluster_latent_mean_decode`; the
diagnostic is `velocity_cluster_image_mean`. Repeating a stage reuses completed
shards. Changed source/settings need a new `MEANFLOW_STORAGE` root and replay audit;
existing run identities are immutable. Preserve each campaign's compilation cache.

For individual stages, use `development`, `score --split development`, `select-k`,
`test`, `score --split test`, and `summarize` with the same `--config` argument,
after the campaign's preparation/audit checks.

## Main results

`results/consensus-main-v4/` contains the original development/test CSVs, their
aggregate table, frozen K selection and recorded configuration. The duplicate
class-name mapping is stored once in `consensus_ablation/imagenet_class_index.json`;
restoring it as `class_mapping` reconstructs the original configuration hash.
Historical paths identify the completed run and are not execution defaults.
Detailed logs, audit records, generated plots, weights and datasets stay outside Git.

Both arms selected K=15 using development KID. Values below are held-out means
over three generation populations after averaging clustering seeds; ± is the
sample SD across those three populations. KID values are raw, not multiplied by 1,000.

| Method | K | Test KID ± generation SD | Test FID |
| --- | ---: | ---: | ---: |
| Single candidate 0 | — | 0.0336015 ± 0.0018320 | 79.8481 |
| Average Image (A) | 15 | 0.1601224 ± 0.0000788 | 244.3934 |
| Average Velocity (B) | 15 | 0.2417244 ± 0.0008042 | 273.1397 |
| B memberships, image mean | 15 | 0.1649789 ± 0.0006647 | 248.3100 |

Both approaches worsen KID relative to ordinary sampling in every generation
population. These subset FIDs are not comparable with published full-ImageNet
FID. KID uses the unbiased cubic polynomial estimator with subset size 500,
100 draws and seed 31415; precision/recall use k=3 Inception neighborhood balls.
Each setting has 500/2,000 generated development/test images per generation seed,
against 2,000/3,000 references. Nine correlated seed combinations are not nine
independent populations; no confidence interval is claimed.

Regenerate quantitative tables and plots from the included CSVs:

```bash
python -m consensus_ablation summarize --recorded-run results/consensus-main-v4 --report-dir work/recorded-report
```

This writes derived CSVs, PNG/PDF plots, `tables.tex` and `summary.md` without
changing the original results. Recomputing image-based scores or the qualitative
panel requires a new campaign or the original feature/audit caches.

The completed run used one B200. Different hardware and fresh compilations can
change floating-point results; the replay gate checks two fresh processes on the
current machine. Full GPU reproduction was not rerun during cleanup; a real
checkpoint CPU smoke run exercised both arms and Inception extraction.

## References and license

Model/solver code comes from [Geng et al., *Mean Flows for One-step Generative
Modeling* (2025)](https://arxiv.org/abs/2505.13447) and its
[official implementation](https://github.com/Gsunshine/meanflow).
The upstream [MIT license](LICENSE) and model-source references are preserved.
Metrics follow [KID](https://arxiv.org/abs/1801.01401),
[improved precision/recall](https://arxiv.org/abs/1904.06991) and
[torch-fidelity 0.3.0](https://github.com/toshas/torch-fidelity/tree/v0.3.0).
Weights, the VAE and ImageNet remain subject to their source terms.
