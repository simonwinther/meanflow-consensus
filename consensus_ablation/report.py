"""Measured tables and figures, with sampling and optimization variation apart."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate import (DIAGNOSTIC_METHOD, IMAGE_METHOD, MAIN_METHODS,
                       VELOCITY_METHOD, atomic_csv, read_csv)
from .reference import atomic_json, config_dict

LABELS = {
    "single_sample": "Single candidate 0",
    IMAGE_METHOD: "A: image clusters, image mean",
    VELOCITY_METHOD: "B: velocity clusters, decode latent mean",
    DIAGNOSTIC_METHOD: "Diagnostic: velocity clusters, image mean",
}
QUANTITIES = ("kid", "fid", "generative_precision", "generative_recall",
              "mean_selected_cluster_size", "mean_selected_cluster_fraction", "mean_cluster_dispersion")


def _number(value):
    return float(value) if value not in (None, "") else None


def _statistics(values):
    values = [float(x) for x in values if x is not None]
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None)


def summarize_seeds(rows):
    """Each generation population gets one mean before between-seed variation."""
    populations = defaultdict(list)
    for row in rows:
        populations[row["split"], row["method"], row["k"], row["generation_seed"]].append(row)
    sensitivity, summaries = [], defaultdict(list)
    for (split, method, k, generation), group in sorted(populations.items()):
        base = {"split": split, "method": method, "k": k, "generation_seed": generation,
                "clustering_seed_count": len(group), "generated_count": group[0]["generated_count"]}
        if len({row["generated_pool_manifest_hash"] for row in group}) != 1:
            raise ValueError("Clustering seeds refer to unmatched generated pools")
        for quantity in QUANTITIES:
            mean, clustering_sd = _statistics([_number(row.get(quantity)) for row in group])
            base[quantity + "_clustering_mean"] = mean
            base[quantity + "_clustering_sd"] = clustering_sd
        base["kid_subset_std_mean"] = _statistics([_number(row["kid_subset_std"]) for row in group])[0]
        sensitivity.append(base)
        summaries[split, method, k].append(base)
    aggregate = []
    for (split, method, k), group in sorted(summaries.items()):
        summary = {"split": split, "method": method, "k": k, "generation_seed_count": len(group)}
        for quantity in QUANTITIES:
            mean, generation_sd = _statistics([row[quantity + "_clustering_mean"] for row in group])
            summary[quantity + "_mean"] = mean
            summary[quantity + "_generation_sd"] = generation_sd
            summary[quantity + "_mean_clustering_sd"] = _statistics([row[quantity + "_clustering_sd"] for row in group])[0]
        summary["kid_subset_std_mean"] = _statistics([row["kid_subset_std_mean"] for row in group])[0]
        aggregate.append(summary)
    return sensitivity, aggregate


def selected_pairwise_comparisons(rows, selection):
    if not selection:
        return []
    choices = selection["selected_k"]
    pairs = [
        (IMAGE_METHOD, str(choices[IMAGE_METHOD]), "single_sample", "", "selected A minus ordinary sampling"),
        (VELOCITY_METHOD, str(choices[VELOCITY_METHOD]), "single_sample", "", "selected B minus ordinary sampling"),
        (IMAGE_METHOD, str(choices[IMAGE_METHOD]), IMAGE_METHOD, "1", "selected A minus all-image mean"),
        (VELOCITY_METHOD, str(choices[VELOCITY_METHOD]), VELOCITY_METHOD, "1", "selected B minus all-latent mean"),
        (VELOCITY_METHOD, str(choices[VELOCITY_METHOD]), DIAGNOSTIC_METHOD, str(choices[VELOCITY_METHOD]), "decoder-side averaging at fixed velocity memberships"),
        (IMAGE_METHOD, str(choices[IMAGE_METHOD]), VELOCITY_METHOD, str(choices[VELOCITY_METHOD]), "separately tuned methods; mechanism effects combined"),
    ]
    # Same-K representation comparisons are a separate mechanism diagnostic.
    common_k = sorted({row["k"] for row in rows if row["method"] == IMAGE_METHOD}, key=int)
    pairs.extend((IMAGE_METHOD, k, DIAGNOSTIC_METHOD, k, "same-K representation comparison with image aggregation") for k in common_k)
    by_setting = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            by_setting[row["method"], row["k"], row["generation_seed"]].append(row)
    comparisons = []
    seeds = sorted({row["generation_seed"] for row in rows if row["split"] == "test"}, key=int)
    for left, kl, right, kr, interpretation in pairs:
        for seed in seeds:
            a, b = by_setting.get((left, kl, seed), []), by_setting.get((right, kr, seed), [])
            if not a or not b:
                continue
            if len({x["generated_pool_manifest_hash"] for x in a + b}) != 1:
                raise ValueError("Selected comparison does not match generated pool IDs")
            km_a, km_b = {x["kmeans_seed"] for x in a}, {x["kmeans_seed"] for x in b}
            if km_a != {""} and km_b != {""} and km_a != km_b:
                raise ValueError("Selected comparison does not match clustering runs")
            row = {"split": "test", "left_method": left, "left_k": kl, "right_method": right,
                   "right_k": kr, "generation_seed": seed, "comparison": interpretation,
                   "interval_method": "none; paired seed-level differences, no bootstrap interval",
                   "generated_pool_manifest_hash": a[0]["generated_pool_manifest_hash"]}
            for metric in ("kid", "fid", "generative_precision", "generative_recall"):
                row[metric + "_left"] = float(np.mean([float(x[metric]) for x in a]))
                row[metric + "_right"] = float(np.mean([float(x[metric]) for x in b]))
                row[metric + "_difference"] = row[metric + "_left"] - row[metric + "_right"]
            comparisons.append(row)
    return comparisons


def _plots_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def method_sketch(directory):
    plt = _plots_module()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    def box(x, y, w, h, label, color="#eef3f7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="#3a4856", lw=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    def arrow(a, b, label=None):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#3a4856"})
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+0.12, label, ha="center", fontsize=9)
    box(.1, 1.85, 2.3, 1.2, "One same-class pool\nεᵢ → uᵢ → zᵢ=εᵢ−uᵢ\nxᵢ=D(zᵢ), i=0,…,N−1")
    box(3.1, 3.35, 2.8, .9, "Full-image K-means\nlargest membership Cₓ", "#dfebff")
    box(7, 3.35, 4.4, .9, "A: mean of xᵢ in Cₓ\nimage mean after decoding", "#dfebff")
    box(3.1, 1.65, 2.8, .9, "Full-velocity K-means\nlargest membership Cᵤ", "#ffe8d5")
    box(7, 1.65, 4.4, .9, "B: D(mean εᵢ − mean uᵢ)\n= D(mean zᵢ), i in Cᵤ", "#ffe8d5")
    box(7, .15, 4.4, .9, "Diagnostic: mean of xᵢ in Cᵤ\nidentical saved velocity memberships", "#e6f0e0")
    arrow((2.4, 2.9), (3, 3.8)); arrow((2.4, 2.35), (3, 2.1))
    arrow((5.9, 3.8), (6.9, 3.8)); arrow((5.9, 2.1), (6.9, 2.1))
    arrow((5.9, 1.85), (6.9, .7))
    ax.text(.1, .35, "Shared candidate IDs, seeds and frozen checkpoint.\nConvert each final output once to uint8.", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"method_sketch.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_figure(fig, path, plt):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def measured_figures(directory, sensitivity, aggregate, selection):
    if not aggregate:
        return []
    plt = _plots_module()
    selected = selection.get("selected_k", {}) if selection else {}
    splits = [s for s in ("development", "test") if any(x["split"] == s for x in aggregate)]
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    colors = {IMAGE_METHOD: "#2166ac", VELOCITY_METHOD: "#b35806", DIAGNOSTIC_METHOD: "#549548"}
    for ax, split in zip(axes[0], splits):
        for method in (IMAGE_METHOD, VELOCITY_METHOD, DIAGNOSTIC_METHOD):
            group = sorted([x for x in aggregate if x["method"] == method and x["split"] == split], key=lambda x: int(x["k"]))
            if not group:
                continue
            ks, means = [int(x["k"]) for x in group], np.array([x["kid_mean"] * 1000 for x in group])
            sds = np.array([(x["kid_generation_sd"] or 0) * 1000 for x in group])
            ax.plot(ks, means, "o-", color=colors[method], label=LABELS[method])
            ax.fill_between(ks, means-sds, means+sds, alpha=.12, color=colors[method])
            seed_rows = [x for x in sensitivity if x["method"] == method and x["split"] == split]
            for seed in sorted({x["generation_seed"] for x in seed_rows}):
                seed_group = sorted([x for x in seed_rows if x["generation_seed"] == seed], key=lambda x: int(x["k"]))
                ax.plot([int(x["k"]) for x in seed_group], [x["kid_clustering_mean"] * 1000 for x in seed_group], color=colors[method], alpha=.25, lw=.7)
            if method in selected:
                point = next((x for x in group if int(x["k"]) == int(selected[method])), None)
                if point:
                    ax.scatter([int(point["k"])], [point["kid_mean"] * 1000], s=150, marker="*", edgecolor="black", color=colors[method], zorder=5)
        baseline = [x["kid_mean"] * 1000 for x in aggregate if x["method"] == "single_sample" and x["split"] == split]
        if baseline:
            ax.axhline(baseline[0], color="#444444", ls="--", label="Single candidate 0")
        ax.set(xlabel="K", ylabel="KID × 1,000 (lower is better)", title=split + (" (fixed grid, descriptive)" if split == "test" else " (selection)"))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    _save_figure(fig, directory / "kid_curves", plt)
    if "test" in splits:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in aggregate:
            if row["split"] != "test":
                continue
            method, k = row["method"], row["k"]
            if method != "single_sample" and k != "1" and int(k) != int(selected.get(method, selected.get(VELOCITY_METHOD, -1))):
                continue
            label = LABELS[method] + (f", K={k}" if k else "")
            ax.errorbar(row["generative_recall_mean"], row["generative_precision_mean"],
                        xerr=row["generative_recall_generation_sd"], yerr=row["generative_precision_generation_sd"],
                        fmt="o", capsize=3, label=label)
        ax.set(xlabel="Generative recall (coverage)", ylabel="Generative precision (fidelity)", title="Held-out fidelity and coverage; ±1 generation-seed SD", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        ax.grid(alpha=.2); ax.legend(fontsize=7)
        _save_figure(fig, directory / "precision_recall", plt)
    fig, ax = plt.subplots(figsize=(7, 4))
    for split in splits:
        for method in MAIN_METHODS:
            group = sorted([x for x in aggregate if x["split"] == split and x["method"] == method and x["mean_selected_cluster_fraction_mean"] is not None], key=lambda x: int(x["k"]))
            if group:
                ax.errorbar([int(x["k"]) for x in group], [x["mean_selected_cluster_fraction_mean"] for x in group],
                            yerr=[x["mean_selected_cluster_fraction_generation_sd"] or 0 for x in group],
                            fmt="o-" if split == "test" else "s--", color=colors[method], capsize=3, label=LABELS[method] + ", " + split)
    ax.set(xlabel="K", ylabel="Selected cluster fraction", ylim=(0, 1.03), title="Occupancy; ±1 generation-seed SD")
    ax.grid(alpha=.2); ax.legend(fontsize=7)
    _save_figure(fig, directory / "cluster_behavior", plt)
    return ["kid_curves", "precision_recall", "cluster_behavior"]


def qualitative_panel(config, directory, selection):
    """Use only the predetermined first-class/first-seed/group-0 audit pool."""
    c = config_dict(config)
    audit = Path(c["output_dir"]) / "audit"
    if not audit.exists():
        return None
    expected_class = int(c["classes"][0])
    expected_seed = int(c.get("generation_seeds", [2026])[0])
    for path in sorted(audit.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if not {"candidate_images", "outputs", "metadata"}.issubset(archive.files):
                continue
            metadata = json.loads(str(archive["metadata"].item()))
            if not metadata:
                continue
            base = metadata[0]
            if (int(base["class_id"]), int(base["generation_seed"]), int(base["group_id"])) != (expected_class, expected_seed, 0):
                continue
            if base.get("split") != "test":
                continue
            candidates, outputs = archive["candidate_images"], archive["outputs"]
        if candidates.dtype != np.uint8 or outputs.dtype != np.uint8:
            raise ValueError("Qualitative audit must store common uint8 display conversion")
        choices = selection.get("selected_k", {}) if selection else {}
        indices = []
        km = int(c.get("kmeans_seeds", [11])[0])
        for method in (IMAGE_METHOD, VELOCITY_METHOD, DIAGNOSTIC_METHOD):
            wanted_k = int(choices.get(method, choices.get(VELOCITY_METHOD, c.get("k_values", [1])[0])))
            found = [i for i, row in enumerate(metadata) if row["method"] == method and int(row["k"]) == wanted_k and (wanted_k == 1 or int(row["kmeans_seed"]) == km)]
            if found:
                indices.append(found[0])
        columns = 8
        count = len(candidates) + len(indices)
        plt = _plots_module()
        fig, axes = plt.subplots(math.ceil(count/columns), columns, figsize=(16, 2.2 * math.ceil(count/columns)), squeeze=False)
        memberships = {metadata[i]["method"]: set(metadata[i].get("selected_candidate_indices", [])) for i in indices}
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < len(candidates):
                ax.imshow(candidates[i], vmin=0, vmax=255)
                tags = (" A" if i in memberships.get(IMAGE_METHOD, set()) else "") + (" B" if i in memberships.get(VELOCITY_METHOD, set()) else "")
                ax.set_title(f"Candidate {i}{tags}", fontsize=8)
            elif i < count:
                index = indices[i-len(candidates)]
                ax.imshow(outputs[index], vmin=0, vmax=255)
                ax.set_title(LABELS[metadata[index]["method"]] + f"\nK={metadata[index]['k']}", fontsize=8)
        fig.suptitle(f"Predetermined test audit: class {expected_class}, generation {expected_seed}, group 0; A/B mark memberships", fontsize=11)
        _save_figure(fig, directory / "matched_qualitative", plt)
        return str(path)
    return None


def _format(value, precision=5):
    return "NA" if value is None else f"{value:.{precision}g}"


def execution_runtime_summary(config, rows):
    """Retain measured stage wall time separately from amortized output rows."""
    c = config_dict(config)
    root = Path(c["output_dir"])
    executions, by_split = [], {}
    for path in sorted(root.glob("execution-*.json")):
        record = json.loads(path.read_text())
        split = path.stem[len("execution-"):].rsplit("-", 1)[0]
        seconds = float(record["seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid execution elapsed time: {path}")
        executions.append({"source": str(path), "split": split, **record})
        summary = by_split.setdefault(split, {"elapsed_seconds": 0., "initialization_seconds": 0.,
                                              "processed_pools": 0, "stage_attempts": 0})
        summary["elapsed_seconds"] += seconds
        summary["initialization_seconds"] += float(record.get("initialization_seconds", 0.))
        summary["processed_pools"] += int(record.get("processed_pools", 0))
        summary["stage_attempts"] += 1
    reference_path = root / "reference" / "timing.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    ledger_path = root.parent / "gpu_budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    metric_by_split = {}
    for split in sorted({row["split"] for row in rows}):
        values = [_number(row.get("metric_seconds")) for row in rows if row["split"] == split]
        metric_by_split[split] = sum(values) if all(x is not None for x in values) else None
    return {
        "schema_version": 1, "run_id": c["run_id"], "config_hash": c["config_hash"],
        "units": "wall-clock seconds; these are not measured GPU busy seconds",
        "run_generation_stages": {
            "elapsed_seconds": sum(x["elapsed_seconds"] for x in by_split.values()) if executions else None,
            "by_split": by_split, "execution_records": executions,
            "scope": "All recorded development/test stage attempts, including initialization, pool generation, aggregation, feature subprocesses and resume checks. Feature time is already included; do not add it again.",
        },
        "run_reference_features": {
            "source": str(reference_path), "record": reference,
            "scope": "Reference extraction timer; extractor construction precedes this timer.",
        },
        "run_metric_calls": {
            "seconds_by_split": metric_by_split,
            "scope": "Sum of measured per-setting KID/FID/precision/recall calls. Shared reference-statistic initialization, evaluation I/O and reporting are excluded; this is not full evaluation stage wall time.",
        },
        "assignment_budget": {
            "source": str(ledger_path),
            "charged_seconds": float(ledger["charged_seconds"]) if ledger else None,
            "ledger_snapshot": ledger,
            "scope": "Cumulative assignment execution ledger across run versions, smoke, benchmark and audits charged to it. It includes this run's charged stages; do not add this value to run timings. It is a conservative execution/allocation budget, not GPU utilization.",
        },
        "output_row_accounting": "runtime_summary.csv rows amortize shared pool core work across outputs. total_seconds = core_seconds + feature_seconds + metric_seconds. They are not standalone method runtimes or the campaign execution total.",
    }


def generate_report(config, destination=None):
    c = config_dict(config)
    source_dir = Path(c["output_dir"]) / "results"
    result_dir = Path(destination) if destination is not None else source_dir
    if destination is not None and result_dir.resolve() == source_dir.resolve():
        raise ValueError('Report destination must differ from archived results')
    figures = result_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("development", "test"):
        path = source_dir / f"{split}_metrics.csv"
        if path.exists():
            rows.extend(read_csv(path))
    selection_path = source_dir / "selected_k.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    sensitivity, aggregate = summarize_seeds(rows)
    atomic_csv(result_dir / "seed_sensitivity.csv", sensitivity, list(sensitivity[0]) if sensitivity else ["split", "method", "k", "generation_seed"])
    atomic_csv(result_dir / "aggregate_metrics.csv", aggregate, list(aggregate[0]) if aggregate else ["split", "method", "k"])
    diagnostics_fields = ["split", "method", "k", "generation_seed", "kmeans_seed", "generated_count", "mean_selected_cluster_size", "mean_selected_cluster_fraction", "mean_cluster_dispersion", "mean_full_pool_dispersion", "winner_tie_frequency", "convergence_failure_count"]
    atomic_csv(result_dir / "cluster_diagnostics.csv", [{key: row.get(key) for key in diagnostics_fields} for row in rows], diagnostics_fields)
    runtime_fields = ["split", "method", "k", "generation_seed", "kmeans_seed", "generated_count", "total_candidate_evaluations", "generator_seconds", "decoder_seconds", "clustering_seconds", "core_seconds", "feature_seconds", "metric_seconds", "total_seconds", "peak_gpu_memory_gib"]
    runtime = []
    for row in rows:
        value = {key: row.get(key) for key in runtime_fields}
        value["intrinsic_candidates_per_final_output"] = 1 if row["method"] == "single_sample" else int(c.get("candidates_per_pool", 64))
        value["cost_accounting"] = "amortized shared pool core + allocated feature time + this population's metric calls; not standalone method runtime or campaign elapsed time; see execution_runtime_summary.json"
        runtime.append(value)
    atomic_csv(result_dir / "runtime_summary.csv", runtime, runtime_fields + ["intrinsic_candidates_per_final_output", "cost_accounting"])
    recorded_runtime = source_dir / "execution_runtime_summary.json"
    execution_runtime = (json.loads(recorded_runtime.read_text())
                         if destination is not None and recorded_runtime.exists()
                         else execution_runtime_summary(c, rows))
    atomic_json(result_dir / "execution_runtime_summary.json", execution_runtime)
    comparisons = selected_pairwise_comparisons(rows, selection)
    atomic_csv(result_dir / "pairwise_selected_comparisons.csv", comparisons, list(comparisons[0]) if comparisons else ["split", "left_method", "right_method", "generation_seed", "kid_difference", "interval_method"])
    method_sketch(figures)
    measured_figures(figures, sensitivity, aggregate, selection)
    audit_path = qualitative_panel(c, figures, selection)
    lines = [f"# MeanFlow inference consensus: {c['run_id']}", "", f"Executed/configured preset: **{c['preset']}**. Configuration hash: `{c['config_hash']}`.", ""]
    if not rows:
        lines += ["No real-reference distribution metrics are available. The experiment has not established a quality improvement or a winning K. Generation and correctness checks can proceed while reference access is resolved.", ""]
    elif c.get("preset") == "smoke":
        lines += ["These tiny smoke metrics validate execution only and are not evidence about model quality. No scientific K selection is made.", ""]
    elif c.get("preset") == "compact":
        lines += ["Compact fallback results are exploratory. The class subset and development population are small; uncertainty limits conclusions.", ""]
    lines += ["## Methods and interpretation", "",
              "The frozen MeanFlow checkpoint supplies one same-class pool for all methods and K values. A clusters every floating decoded-image coordinate and averages images in the largest cluster. B clusters full predicted average velocities and decodes the mean endpoint latent from the matching noise–velocity pairs. The diagnostic averages decoded images over exactly B's saved memberships.", "",
              "K controls a partition, not a quality detector. The largest cluster is the most frequent group under the fixed representation and fitting rule. Pixel averaging can blur distinct valid images; smaller selected groups may appear sharper simply because fewer images are averaged. Velocity differences mix starting-noise and endpoint differences. For fixed members, mean(noise) − mean(velocity) equals mean(endpoint latent); the nonlinear VAE makes decoding that mean different from averaging decoded images.", "",
              f"Consensus consumes {c.get('candidates_per_pool', 64)} candidate evaluations per final output; ordinary sampling intrinsically consumes one. Pools are reused throughout this campaign. Scores concern this fixed class subset and checkpoint; subset FID is not comparable with published full-ImageNet FID.", "",
              "KID uses the unbiased polynomial estimator with kernel ((a·b)/d+1)^3; same-set diagonals are excluded and negative estimates retained. Its subset-draw SD describes estimator variation conditional on the sampled dataset. Generative precision/recall implement the published union-of-neighborhood-balls definition with the declared Inception features and k=3 (or the recorded fixed configuration).", "",
              "Generation-seed means first average the clustering seeds. Generation SD and within-generation clustering SD are reported separately; nine seed combinations are not nine independent populations. Selected comparisons use paired seed-level differences with matching pool IDs. No bootstrap confidence interval is claimed.", ""]
    if selection:
        lines += ["## Frozen development selection", "", "K was selected separately per arm by development KID, averaging clustering seeds within generation seeds and then averaging generation populations. Ties within 1e-12 raw KID choose smaller K. Held-out K curves describe the fixed preregistered grid and cannot replace these choices.", ""]
        for method in MAIN_METHODS:
            k = selection["selected_k"][method]
            score = selection["scores"][method][str(k)]["mean_development_kid"]
            lines += [f"- {LABELS[method]}: K={k}, development KID={score:.7g}." + (" Clustering did not improve the development criterion within this grid." if k == 1 else "")]
        lines += [""]
    if aggregate:
        lines += ["## Measured results", "", "The ± value is one generation-seed SD after averaging clustering runs; the separate clustering SD is the mean within-generation SD. A single generation seed has no estimated generation SD.", "",
                  "| Split | Method | K | KID | Generation SD | Clustering SD | FID | Precision | Recall |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in aggregate:
            lines += [f"| {row['split']} | {LABELS[row['method']]} | {row['k'] or 'NA'} | {_format(row['kid_mean'])} | {_format(row['kid_generation_sd'])} | {_format(row['kid_mean_clustering_sd'])} | {_format(row['fid_mean'])} | {_format(row['generative_precision_mean'])} | {_format(row['generative_recall_mean'])} |"]
        lines += ["", "Differences inconsistent across generation seeds remain inconclusive. Tuned A-versus-B comparisons combine representation and decoder-side effects; use same-K A-versus-diagnostic and identical-membership B-versus-diagnostic comparisons for interpretation. Negative or mixed results do not justify changing the predefined protocol.", ""]
    if comparisons:
        lines += ["## Paired held-out interpretation", ""]
        for method in MAIN_METHODS:
            differences = [row["kid_difference"] for row in comparisons if row["left_method"] == method and row["right_method"] == "single_sample"]
            if differences:
                direction = "lower in every generation population" if all(x < 0 for x in differences) else "higher in every generation population" if all(x > 0 for x in differences) else "inconsistent across generation populations"
                lines += [f"- Selected {('A' if method == IMAGE_METHOD else 'B')} versus ordinary sampling: KID is {direction}; paired differences are {', '.join(f'{x:.7g}' for x in differences)}. No confidence interval is inferred from three populations."]
        diagnostic = [row for row in comparisons if row["comparison"] == "decoder-side averaging at fixed velocity memberships"]
        if diagnostic:
            diagnostic_differences = ", ".join(format(x["kid_difference"], ".7g") for x in diagnostic)
            lines += [f"- B minus its identical-membership image-mean diagnostic: paired KID differences {diagnostic_differences}. This measures the decoder-side aggregation change, not improved velocity-cluster selection."]
        lines += [""]
    if execution_runtime['run_generation_stages']['elapsed_seconds'] is not None:
        lines += ["## Execution accounting", "",
                  f"Recorded development/test generation-stage elapsed time for this run: {_format(execution_runtime['run_generation_stages']['elapsed_seconds'])} seconds. This includes initialization and feature subprocesses across all recorded attempts. Cumulative assignment ledger at report time: {_format(execution_runtime['assignment_budget']['charged_seconds'])} seconds, covering charged prior versions, smoke, benchmark and audit work as well as this run. These overlapping totals must not be added. `execution_runtime_summary.json` retains the source execution records, reference timer and assignment-ledger snapshot.", "",
                  "Per-setting runtime rows amortize shared generation/aggregation core work across outputs. Their total_seconds includes core_seconds, feature_seconds and metric_seconds. They do not measure a method run alone, and they exclude shared feature startup, reference statistics, evaluation I/O and reporting overhead. The stage records and cumulative ledger supply execution accounting; resource timers are wall time, not GPU busy time.", ""]
    lines += ["## Artifacts and limits", "", "CSV tables retain raw scores, exact sample counts, seed IDs, manifest hashes and resource fields. Blank resource fields mean unmeasured, not zero. New scoring runs save exact KID subset schedules and freeze the development selection.", ""]
    lines += [f"Matched qualitative panel source: `{audit_path}`." if audit_path else "No eligible predetermined held-out qualitative audit was found; no synthetic or cherry-picked panel was substituted.", "",
              "References: [MeanFlow](https://arxiv.org/abs/2505.13447), [KID](https://arxiv.org/abs/1801.01401), [precision/recall](https://arxiv.org/abs/1904.06991). KID's implementation definition follows [torch-fidelity 0.3.0](https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py); the NumPy manifold implementation follows the mathematical definition of the [published metric](https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py).", ""]
    (result_dir / "summary.md").write_text("\n".join(lines))
    latex = [r"\begin{tabular}{lllrrrrr}", r"Split & Method & K & KID & Gen. SD & Cluster SD & FID & Recall \\", r"\hline"]
    for row in aggregate:
        method = {IMAGE_METHOD: "A", VELOCITY_METHOD: "B", DIAGNOSTIC_METHOD: "Diagnostic", "single_sample": "Single"}[row["method"]]
        latex.append(" & ".join([row["split"], method, row["k"] or "NA", _format(row["kid_mean"]), _format(row["kid_generation_sd"]), _format(row["kid_mean_clustering_sd"]), _format(row["fid_mean"]), _format(row["generative_recall_mean"])]) + r" \\")
    latex += [r"\end{tabular}", "% Raw KID; Gen. SD after averaging clustering seeds; no confidence interval."]
    (result_dir / "tables.tex").write_text("\n".join(latex) + "\n")
    return result_dir / "summary.md"
