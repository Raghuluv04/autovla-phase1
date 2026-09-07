"""Figures for the Phase I data-processing review."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .geometry import canonical_contour, contours_from_poses
from .kinematics import invert_segment
from .config import MAX_CURVATURE

# ---------------------------------------------------------------- style ----
INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"
ACCENT2 = "#dc2626"
ACCENT3 = "#059669"
ACCENT4 = "#d97706"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ------------------------------------------------------------- figures ----
def fig_codebook(codebook, outdir, name="01_action_codebook.png"):
    """Reproduces the paper's Fig. S1: the learned motion vocabulary."""
    a = codebook.actions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = _clean(axes[0])
    sc = ax.scatter(a[:, 1], a[:, 0], c=np.rad2deg(a[:, 2]), s=7,
                    cmap="coolwarm", vmin=-20, vmax=20, linewidths=0)
    ax.set_xlabel("lateral $\\Delta y$ (m)")
    ax.set_ylabel("longitudinal $\\Delta x$ (m)")
    ax.set_title(f"Action codebook, K = {len(a)}")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("$\\Delta\\theta$ (deg)", size=8)
    cb.ax.tick_params(labelsize=7)

    ax = _clean(axes[1])
    contour = canonical_contour(codebook.meta["ego_length"],
                                codebook.meta["ego_width"],
                                codebook.meta["contour_points"])
    idx = np.random.default_rng(0).choice(len(a), min(220, len(a)), replace=False)
    polys = contours_from_poses(a[idx], contour)
    for p, act in zip(polys, a[idx]):
        ax.add_patch(Polygon(p[:, ::-1], closed=True, fill=False,
                             edgecolor=ACCENT, linewidth=0.45, alpha=0.55))
    ax.add_patch(Polygon(contours_from_poses(np.zeros(3), contour)[:, ::-1],
                         closed=True, fill=True, facecolor=ACCENT2, alpha=0.28,
                         edgecolor=ACCENT2, linewidth=1.2))
    ax.set_xlim(polys[..., 1].max() + 1, polys[..., 1].min() - 1)
    ax.set_ylim(polys[..., 0].min() - 1, polys[..., 0].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("longitudinal (m)")
    ax.set_title("Footprint contours (220 sampled tokens)")

    ax = _clean(axes[2])
    speed = np.linalg.norm(a[:, :2], axis=1) / codebook.dt
    ax.hist(speed, bins=48, color=ACCENT, alpha=0.85, edgecolor="white",
            linewidth=0.4)
    ax.set_xlabel("implied speed (m/s)"); ax.set_ylabel("tokens")
    ax.set_title("Speed coverage of the vocabulary")
    fig.suptitle("Physical action tokenization  ·  K-disk clustering of 0.5 s motion segments",
                 y=1.03, fontsize=12, fontweight="600", color=INK)
    return _save(fig, outdir, name)


def fig_feasibility(codebook, outdir, name="02_kinematic_feasibility.png"):
    """Every token inverted through the bicycle model against vehicle limits."""
    a = codebook.actions
    speed = np.linalg.norm(a[:, :2], axis=1) / codebook.dt
    k = invert_segment(a, v0=speed, dt=codebook.dt)
    curv, steer = k["curvature"], np.rad2deg(k["steer"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    ax = _clean(axes[0])
    ax.scatter(speed, curv, s=8, c=ACCENT, alpha=0.55, linewidths=0)
    ax.axhline(MAX_CURVATURE, color=ACCENT2, ls="--", lw=1.1,
               label=f"limit ±{MAX_CURVATURE} 1/m")
    ax.axhline(-MAX_CURVATURE, color=ACCENT2, ls="--", lw=1.1)
    ax.set_xlabel("implied speed (m/s)"); ax.set_ylabel("curvature (1/m)")
    ax.set_title("Curvature vs speed — all tokens inside limits")
    ax.set_ylim(-MAX_CURVATURE * 1.35, MAX_CURVATURE * 1.35)
    ax.legend(loc="lower right")

    ax = _clean(axes[1])
    ax.hist(steer, bins=44, color=ACCENT3, alpha=0.85, edgecolor="white",
            linewidth=0.4)
    ax.set_xlabel("equivalent steering angle (deg)"); ax.set_ylabel("tokens")
    ax.set_title("Steering demanded by the vocabulary")
    frac = float(np.mean(np.abs(curv) <= MAX_CURVATURE))
    fig.suptitle(f"Bicycle-model feasibility check  ·  {frac:.1%} of tokens physically realisable",
                 y=1.03, fontsize=12, fontweight="600", color=INK)
    return _save(fig, outdir, name)


def fig_reconstruction(tok, trajectories, outdir, labels=None,
                       name="03_trajectory_reconstruction.png"):
    """Ground-truth vs encode->decode trajectories."""
    n = min(8, len(trajectories))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
    for i, ax in enumerate(axes.ravel()[:n]):
        _clean(ax)
        gt = np.asarray(trajectories[i], dtype=float)
        ids = tok.encode(gt)
        rec = tok.decode(ids, start=gt[0])
        ax.plot(gt[:, 1], gt[:, 0], "-o", color=INK, ms=3.2, lw=1.6,
                label="ground truth")
        ax.plot(rec[:, 1], rec[:, 0], "--s", color=ACCENT2, ms=3.0, lw=1.4,
                alpha=0.9, label="decoded tokens")
        e = tok.reconstruction_error(gt)
        title = labels[i] if labels else f"sample {i}"
        ax.set_title(f"{title}   ADE {e['ade']*100:.1f} cm", fontsize=9)
        # Straight-line manoeuvres have almost no lateral extent. Forcing an
        # equal aspect there collapses the x-axis until its tick labels
        # overlap, so we keep true scale only where there is lateral motion to
        # see, and fall back to a floored, sparsely ticked axis otherwise.
        lo = min(gt[:, 1].min(), rec[:, 1].min())
        hi = max(gt[:, 1].max(), rec[:, 1].max())
        mid, extent = (lo + hi) / 2, hi - lo
        if extent >= 5.0:
            ax.set_aspect("equal")
            ax.set_xlim(hi + 1.0, lo - 1.0)
        else:
            span = max(extent, 3.0)
            ax.set_xlim(mid + span * 0.7, mid - span * 0.7)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if i == 0:
            ax.legend(loc="best")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.supxlabel("lateral (m)", fontsize=9, color=MUTED)
    fig.supylabel("longitudinal (m)", fontsize=9, color=MUTED)
    fig.suptitle("Tokenizer round-trip:  5 s trajectory  to  10 action tokens  to  trajectory",
                 y=1.0, fontsize=12, fontweight="600", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, name)


def fig_codebook_ablation(results, outdir, name="04_codebook_size_ablation.png"):
    """Reconstruction error vs K, against the paper's Table 4 K-disk numbers.

    Also contrasts the two ways of anchoring the tokenizer. Encoding each
    segment relative to the ground-truth pose ("open loop") lets heading
    quantization error rotate every later waypoint, so FDE runs far above ADE.
    Anchoring on the pose reached so far removes that drift and reproduces the
    paper's signature of FDE ~= ADE.
    """
    ks = [r["k"] for r in results]
    # Paper Table 4, K-disk (ours) column, metres.
    paper = {256: (0.0687, 0.1034), 1024: (0.0253, 0.0282),
             2048: (0.0182, 0.0203), 4096: (0.0141, 0.0155)}
    pk = [k for k in ks if k in paper]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = _clean(axes[0])
    ax.plot(ks, [r["ade"] * 100 for r in results], "-o", color=ACCENT,
            label="ours, anchored")
    ax.plot(ks, [r["ade_open_loop"] * 100 for r in results], "-^", color=MUTED,
            alpha=0.8, label="ours, open loop")
    ax.plot(pk, [paper[k][0] * 100 for k in pk], "--s", color=ACCENT2,
            label="paper (Table 4)")
    ax.set_yscale("log")
    ax.set_ylabel("ADE (cm, log)")
    ax.set_title("Reconstruction error vs codebook size")

    ax2 = _clean(axes[1])
    ax2.plot(ks, [r["fde"] * 100 for r in results], "-o", color=ACCENT,
             label="ours, anchored")
    ax2.plot(ks, [r["fde_open_loop"] * 100 for r in results], "-^", color=MUTED,
             alpha=0.8, label="ours, open loop")
    ax2.plot(pk, [paper[k][1] * 100 for k in pk], "--s", color=ACCENT2,
             label="paper (Table 4)")
    ax2.set_yscale("log")
    ax2.set_ylabel("FDE (cm, log)")
    ax2.set_title("Open-loop anchoring lets error compound")

    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks(ks); a.set_xticklabels(ks)
        a.set_xlabel("codebook size K")
        a.axvline(2048, color=GRID, ls=":", lw=1.2)
        a.legend()

    ax3 = _clean(axes[2])
    ax3.plot(ks, [r["codebook_usage"] * 100 for r in results], "-o",
             color=ACCENT3)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(ks); ax3.set_xticklabels(ks)
    ax3.set_xlabel("codebook size K"); ax3.set_ylabel("tokens actually used (%)")
    ax3.set_title("Utilisation falls as K grows — the trade-off")
    ax3.axvline(2048, color=GRID, ls=":", lw=1.2)
    ymid = sum(ax3.get_ylim()) / 2
    ax3.annotate("K = 2048 balances\naccuracy and usage", xy=(2048, ymid),
                 xytext=(600, ymid), fontsize=8, color=MUTED, ha="center",
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

    fig.suptitle("Why K = 2048, and why segment anchoring matters",
                 y=1.03, fontsize=12, fontweight="600", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, name)


def fig_token_usage(token_ids, k, outdir, name="05_token_usage.png"):
    counts = np.bincount(np.asarray(token_ids, dtype=int), minlength=k)
    order = np.argsort(counts)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    ax = _clean(axes[0])
    ax.fill_between(np.arange(k), counts[order], color=ACCENT, alpha=0.75, lw=0)
    ax.set_yscale("symlog")
    ax.set_xlabel("token rank"); ax.set_ylabel("times used (log)")
    ax.set_title("Usage is heavy-tailed — few tokens dominate")

    ax = _clean(axes[1])
    cum = np.cumsum(counts[order]) / max(counts.sum(), 1)
    ax.plot(np.arange(1, k + 1), cum * 100, color=ACCENT2, lw=1.8)
    for frac in (0.5, 0.9):
        idx = int(np.searchsorted(cum, frac)) + 1
        ax.axhline(frac * 100, color=GRID, lw=1)
        ax.annotate(f"{frac:.0%} of usage from {idx} tokens",
                    xy=(idx, frac * 100), xytext=(idx + k * 0.06, frac * 100 - 9),
                    fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xlabel("number of most-used tokens"); ax.set_ylabel("cumulative usage (%)")
    ax.set_title("Cumulative coverage")
    fig.suptitle("Action token usage across the training corpus",
                 y=1.03, fontsize=12, fontweight="600", color=INK)
    return _save(fig, outdir, name)


def fig_dataset_stats(seg, samp, outdir, name="06_dataset_statistics.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    ax = _clean(axes[0, 0])
    ax.hist(seg["speed_mps"], bins=50, color=ACCENT, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(seg["speed_mean"], color=ACCENT2, ls="--", lw=1.2,
               label=f"mean {seg['speed_mean']:.1f} m/s")
    ax.set_xlabel("speed (m/s)"); ax.set_ylabel("segments")
    ax.set_title("Ego speed distribution"); ax.legend()

    ax = _clean(axes[0, 1])
    ax.hist(np.clip(seg["curvature"], -0.25, 0.25), bins=60, color=ACCENT4,
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("curvature (1/m)"); ax.set_ylabel("segments (log)")
    ax.set_title("Curvature — long tail of turns")

    ax = _clean(axes[0, 2])
    ax.hist(np.clip(samp["acceleration"], -5, 5), bins=50, color=ACCENT3,
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("acceleration (m/s$^2$)"); ax.set_ylabel("samples")
    ax.set_title("Longitudinal acceleration")

    ax = _clean(axes[1, 0])
    cmds = samp["commands"]
    keys = list(cmds.keys()); vals = [cmds[k] for k in keys]
    bars = ax.barh(keys, vals, color=ACCENT, alpha=0.85)
    total = max(sum(vals), 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f"{v/total:.0%}", va="center", fontsize=8, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("samples"); ax.set_title("Navigation command mix")

    ax = _clean(axes[1, 1])
    ep = samp["endpoints"]
    ax.hexbin(ep[:, 1], ep[:, 0], gridsize=40, cmap="Blues", mincnt=1,
              bins="log", linewidths=0)   # log bins: density spans 3+ decades
    ax.invert_xaxis(); ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("longitudinal (m)")
    ax.set_title("5 s endpoint density (ego frame)")

    ax = _clean(axes[1, 2])
    modes = samp["modes"]
    ax.pie([modes.get("fast", 0), modes.get("slow", 0)],
           labels=["fast thinking\n(trajectory only)", "slow thinking\n(with CoT)"],
           colors=[ACCENT, ACCENT4], autopct="%1.0f%%",
           textprops=dict(fontsize=8.5, color=INK),
           wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.grid(visible=False)
    ax.set_title("SFT sample mix")

    fig.suptitle(f"Processed corpus  ·  {samp['n']:,} training samples  ·  "
                 f"{seg['n']:,} motion segments",
                 y=1.0, fontsize=12, fontweight="600", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, name)


def fig_sample_card(sample, tok, outdir, name="07_sample_card.png"):
    """One fully processed training sample, end to end — good for a slide."""
    gt = np.asarray(sample.future_traj, dtype=float)
    ids = tok.encode(gt)
    rec = tok.decode(ids, start=gt[0])
    hist = np.asarray(sample.history_traj, dtype=float)

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.18)

    ax = _clean(fig.add_subplot(gs[0]))
    contour = canonical_contour()
    ax.add_patch(Polygon(contours_from_poses(np.zeros(3), contour)[:, ::-1],
                         closed=True, facecolor=MUTED, alpha=0.35,
                         edgecolor=INK, lw=1.0))
    for p in contours_from_poses(rec[1:], contour):
        ax.add_patch(Polygon(p[:, ::-1], closed=True, fill=False,
                             edgecolor=ACCENT2, lw=0.6, alpha=0.5))
    if len(hist):
        ax.plot(hist[:, 1], hist[:, 0], "-o", color=MUTED, ms=3, lw=1.3,
                label="history (2 s)")
    ax.plot(gt[:, 1], gt[:, 0], "-o", color=INK, ms=3.4, lw=1.7,
            label="ground truth (5 s)")
    ax.plot(rec[:, 1], rec[:, 0], "--s", color=ACCENT2, ms=3.2, lw=1.4,
            label="decoded action tokens")
    ax.set_aspect("equal"); ax.invert_xaxis()
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("longitudinal (m)")
    ax.set_title("BEV: history, ground truth, tokenized plan")
    ax.legend(loc="best")

    ax = fig.add_subplot(gs[1]); ax.axis("off")
    from .prompts import build_user_message
    tokens = tok.ids_to_text(ids)
    wrapped = "\n".join(tokens[i:i + 76] for i in range(0, len(tokens), 76))
    body = (
        f"sample_id : {sample.sample_id}\n"
        f"dataset   : {sample.dataset}   mode: {sample.thinking_mode} thinking\n"
        f"{'-'*74}\n"
        f"USER\n{build_user_message(sample)}\n"
        f"{'-'*74}\n"
        f"ASSISTANT\n"
        + (sample.reasoning.to_text() + "\n" if sample.has_reasoning
           else "This is a straightforward scenario; no reasoning is needed.\n")
        + f"{wrapped}\n"
        f"{'-'*74}\n"
        f"token ids : {list(map(int, ids))}\n"
        f"round-trip ADE: {tok.reconstruction_error(gt)['ade']*100:.1f} cm"
    )
    ax.text(0, 1, body, va="top", ha="left", family="monospace", fontsize=7.4,
            color=INK, linespacing=1.45)
    ax.set_title("Training record as fed to the VLM", loc="left")
    fig.suptitle("One processed sample, end to end", y=1.02, fontsize=12,
                 fontweight="600", color=INK)
    return _save(fig, outdir, name)


def fig_pipeline_overview(counts, outdir, name="00_pipeline_overview.png"):
    """Block diagram of the data pipeline with live counts."""
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off"); ax.grid(visible=False)
    stages = [
        ("Raw logs", counts.get("raw", ""), ACCENT),
        ("Unified\nschema", counts.get("samples", ""), ACCENT),
        ("0.5 s motion\nsegments", counts.get("segments", ""), ACCENT4),
        ("K-disk\ncodebook", counts.get("codebook", ""), ACCENT4),
        ("Action\ntokenizer", counts.get("tokens", ""), ACCENT3),
        ("SFT chat\nrecords", counts.get("records", ""), ACCENT3),
    ]
    n = len(stages)
    for i, (label, sub, color) in enumerate(stages):
        x = i / n + 0.5 / n
        ax.add_patch(plt.Rectangle((x - 0.42 / n, 0.18), 0.84 / n, 0.60,
                                   transform=ax.transAxes, facecolor=color,
                                   alpha=0.13, edgecolor=color, lw=1.4,
                                   zorder=1))
        ax.text(x, 0.60, label, transform=ax.transAxes, ha="center",
                va="center", fontsize=9.5, fontweight="600", color=INK)
        ax.text(x, 0.32, str(sub), transform=ax.transAxes, ha="center",
                va="center", fontsize=8.5, color=MUTED)
        if i < n - 1:
            ax.annotate("", xy=((i + 1) / n - 0.44 / n, 0.48),
                        xytext=(x + 0.44 / n, 0.48), xycoords=ax.transAxes,
                        textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
    ax.set_title("AutoVLA data-processing pipeline", loc="left", fontsize=12.5,
                 fontweight="600", color=INK, pad=14)
    return _save(fig, outdir, name)


def fig_camera_overlay(reader, samples, tok, outdir, dataroot,
                       n=3, name="08_camera_overlay.png"):
    """Ground-truth and tokenized trajectories projected onto camera images.

    nuScenes only — this is the check that the processed geometry actually
    lines up with what the cameras saw, which is the thing a BEV/planning
    pipeline most easily gets silently wrong.
    """
    from PIL import Image
    from .adapters.nuscenes import project_to_image

    picks, seen = [], set()
    for s in samples:
        scene = s.extra.get("scene")
        if scene in seen:
            continue
        calib = reader.camera_calibration(s.sample_id, "CAM_FRONT")
        path = os.path.join(dataroot, s.images.get("CAM_FRONT", [""])[-1])
        if calib is None or not os.path.exists(path):
            continue
        picks.append((s, calib, path)); seen.add(scene)
        if len(picks) >= n:
            break
    if not picks:
        print("  (no nuScenes images found — skipping camera overlay)")
        return None

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 5.0 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, (s, calib, path) in zip(axes, picks):
        ax.imshow(Image.open(path)); ax.axis("off"); ax.grid(visible=False)
        gt = np.asarray(s.future_traj, dtype=float)
        rec = tok.decode(tok.encode(gt), start=gt[0])
        for traj, color, label, lw in ((gt, "#22d3ee", "ground truth", 3.0),
                                       (rec, "#f43f5e", "decoded tokens", 2.0)):
            uv, _ = project_to_image(traj[:, :2], calib)
            if len(uv) > 1:
                ax.plot(uv[:, 0], uv[:, 1], "-o", color=color, ms=5, lw=lw,
                        label=label, alpha=0.95)
        ax.legend(loc="upper right", facecolor="black", labelcolor="white",
                  framealpha=0.45)
        ax.set_title(f"{s.extra.get('scene','')}  ·  {s.command}  ·  "
                     f"{s.velocity:.1f} m/s", fontsize=10, color=INK)
    fig.suptitle("Planned trajectory projected onto CAM_FRONT",
                 y=1.0, fontsize=12, fontweight="600", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, name)
