# MeanFlow: Average Image and Average Velocity

We compared two ways of averaging samples from the released MeanFlow MF-B/4
EMA checkpoint on ImageNet at 256 × 256 resolution:

- Average Image (A) clusters decoded images using all floating-point pixel
  values, then averages the images in the largest cluster.
- Average Velocity (B) clusters predicted average velocities using all
  coordinates. It averages the matching noise and velocity pairs in the largest
  cluster, then decodes the mean endpoint latent.

For each candidate, `z = epsilon - u`. B computes
`D(mean(epsilon[C]) - mean(u[C]))`; A averages `D(z[C])` using image memberships.
We also average images using B's memberships to check the nonlinear decoder's
effect. All reductions happen before clipping and uint8 conversion. The ordinary
sampling control uses candidate 0. Neither method trains new parameters.

## Installation

Use Linux, [uv](https://docs.astral.sh/uv/) and Python 3.12. The completed run used
Python 3.12.3; installation and CPU execution were also checked on 3.12.13.
Inference uses JAX/jaxlib 0.8.0 with CUDA 13, Flax 0.12.0 and Diffusers 0.35.1.
Feature extraction has a separate environment with PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128 and torch-fidelity 0.3.0. Package versions are pinned in
`scripts/consensus/environment-*.lock.txt`.

Run from the repository root:

```bash
bash scripts/consensus/setup.sh
source work/meanflow-env/bin/activate
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
python -m consensus_ablation inspect --config configs/consensus_main.yaml
```

The CA-bundle path above was tested on Ubuntu/NixOS; use your system's location
if it differs. Environments, assets and runs go under `work/` by default.
For another location, set `MEANFLOW_STORAGE` before installation and in later shells.

Generation needs one visible NVIDIA GPU with compatible drivers. You can
regenerate reports from the included CSVs without a GPU, model weights or data.

## Checkpoint and ImageNet

```bash
python -m consensus_ablation prepare-assets --config configs/consensus_main.yaml
python -m consensus_ablation prepare-data --config configs/consensus_main.yaml
```

Asset preparation downloads the [official MF-B/4 checkpoint](https://drive.google.com/file/d/19MR1WLycyqc627gsOJF4yzR4ZkU8fvOu/view)
and the [Flax VAE](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax/tree/7581b0a0489cc8483c876a728b830b9ce087cf03).
The archive, `b4_240ep_kappa0.5.zip`, is 1,939,631,766 bytes and holds checkpoint
step 1,200,960. Its SHA-256 must match
`cfe35729ecf7dbf6aa4afc51933444a2a45886ce60a55453489e1e5c3b8ffd3c`.
The VAE revision is pinned to `7581b0a0489cc8483c876a728b830b9ce087cf03`.
Inference loads `ema_params`; the Inception weights download on first use.

If the Drive download fails, place the authorized ZIP at
`work/meanflow-assets/b4_240ep_kappa0.5.zip` and rerun asset preparation.

Data preparation needs the ImageNet ILSVRC2012 validation set and devkit.
Obtain access through [ImageNet](https://image-net.org/download.php) under its
data terms. If the download is unavailable, put the authorized archives here
and rerun preparation:

```text
work/imagenet/ILSVRC2012_img_val.tar
work/imagenet/ILSVRC2012_devkit_t12.tar.gz
```

The script checks the published MD5 values, translates labels through the devkit
and extracts images to
`work/imagenet/val/<synset>/ILSVRC2012_val_<index>.JPEG`.
It selects 100 classes with `default_rng(1729)`, keeping 50 images per class.
Class-local hashes of seed 2718 split these into 20 development and 30 test
references per class. References use RGB conversion and ADM-style resizing
and center cropping to 256 × 256. You do not need training images or
full-ImageNet FID statistics.

To train the checkpoint from scratch, use the [original training source](https://github.com/Gsunshine/meanflow/tree/d70cb55d298ee03c53bf6da67bec281082e4e2d9).
That workflow has not been tested locally for this project.

## Run both approaches

After preparing the assets and data, run the campaign on an assigned GPU:

```bash
python -m consensus_ablation campaign --config configs/consensus_main.yaml
```

The campaign starts with the numerical replay check, smoke test and runtime
benchmark. It extracts reference features, generates and scores development
samples, selects K separately for each method, then evaluates held-out samples
and writes the reports. Both methods and their controls share candidate pools.

The benchmark chooses the main protocol if it fits the 48-hour execution budget,
or the compact fallback otherwise. Check the saved decision; compact results
belong to a different protocol. The GPU must already be assigned to your process.

The main protocol uses 100 classes and N=64 candidates per pool. We test
K=[1,2,5,10,15] with generation seeds [2026,2027,2028] and clustering seeds
[11,29,47]. For each class and generation seed, there are 5 development pools
and 20 test pools, giving 480,000 candidate forwards in total.

K-means uses every coordinate and Euclidean distance, with ten K-means++ restarts
and at most 100 Lloyd iterations. The fit with minimum inertia is retained.
Largest-cluster ties use dispersion, then candidate IDs. K=1 averages the whole
pool without repeated clustering runs. Compact uses the first 20 classes and
2/10 development/test pools.

Runs write candidate shards, memberships, features, timings, audit images and
`results/` tables/figures under `work/meanflow-runs/<run_id>/`.
The method names are `image_cluster_image_mean` for A,
`velocity_cluster_latent_mean_decode` for B, and `velocity_cluster_image_mean`
for the diagnostic. Repeating a stage reuses completed shards. Keep each campaign's
compilation cache. Changes to source or settings need a new `MEANFLOW_STORAGE`
root and replay audit; existing run identities are immutable.

You can also run individual stages: `development`, `score --split development`,
`select-k`, `test`, `score --split test`, and `summarize`. Use the same `--config`
argument after completing the campaign's preparation and audit checks.

## Main results

`results/consensus-main-v4/` holds the original development/test CSVs, aggregate
table, frozen K choices and run settings. Class names are stored once in
`consensus_ablation/imagenet_class_index.json`.
Restore that mapping as `class_mapping` to reconstruct the original
configuration hash. Machine paths in these records describe the completed run;
new runs use your current settings. Logs, audit records, generated plots, weights
and datasets are excluded from Git.

Development KID selected K=15 for both methods. For test results, we first average
clustering seeds within each generation population, then average the three
populations. The ± values are their sample SDs. KID uses raw units, without
multiplication by 1,000.

| Method | K | Test KID ± generation SD | Test FID |
| --- | ---: | ---: | ---: |
| Single candidate 0 | — | 0.0336015 ± 0.0018320 | 79.8481 |
| Average Image (A) | 15 | 0.1601224 ± 0.0000788 | 244.3934 |
| Average Velocity (B) | 15 | 0.2417244 ± 0.0008042 | 273.1397 |
| B memberships, image mean | 15 | 0.1649789 ± 0.0006647 | 248.3100 |

Both averaging methods gave worse KID than ordinary sampling in every
generation population. These FIDs use a subset of ImageNet and cannot be
compared with published full-ImageNet FID.

KID uses the unbiased cubic polynomial estimator with subset size 500,
100 draws and seed 31415. Precision/recall use k=3 Inception neighborhood balls.
Each setting has 500/2,000 generated development/test images per generation
seed, compared with 2,000/3,000 references. The nine seed combinations share
generation pools, so there are only three independent generation populations.
We report seed SDs and do not estimate confidence intervals.

To regenerate the quantitative tables and plots from the included CSVs:

```bash
python -m consensus_ablation summarize --recorded-run results/consensus-main-v4 --report-dir work/recorded-report
```

This produces derived CSVs, PNG/PDF plots, `tables.tex` and `summary.md`, leaving
the original results intact. Image-based scoring and the qualitative panel need
a new campaign or the original feature/audit caches.

The completed experiment used one B200. Floating-point results can change
with hardware or fresh compilations, so the replay check compares two fresh
processes on the current machine. A real checkpoint CPU smoke run exercised
both methods and Inception extraction. The full GPU experiment has not been
rerun on this cleaned checkout.

## References and license

The model and solver come from [Geng et al., *Mean Flows for One-step Generative
Modeling* (2025)](https://arxiv.org/abs/2505.13447) and the
[official implementation](https://github.com/Gsunshine/meanflow).
We retain the upstream [MIT license](LICENSE) and model-source references.
Metric definitions follow [KID](https://arxiv.org/abs/1801.01401),
[improved precision/recall](https://arxiv.org/abs/1904.06991) and
[torch-fidelity 0.3.0](https://github.com/toshas/torch-fidelity/tree/v0.3.0).
The weights, VAE and ImageNet have their own source terms.
