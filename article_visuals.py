"""
Generate publication-quality visuals for Medium article on Mamba-2 autoresearch.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("results.tsv", sep="\t")
df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

valid = df[(df["status"] != "CRASH") & (df["val_bpb"] > 0)].copy().reset_index(drop=True)
baseline_bpb = valid.loc[0, "val_bpb"]

kept = valid[valid["status"] == "KEEP"].copy()
disc = valid[valid["status"] == "DISCARD"].copy()
running_min = kept["val_bpb"].cummin()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main progress chart (hero image) — with per-kept-experiment labels
# ═══════════════════════════════════════════════════════════════════════════

# ── Label table: (description_match, label, x_off, y_off) ─────────────
# Hand-tuned offsets to avoid all overlaps.  Positive y = above point.
LABELS = [
    # Phase 1: Hyperparams — spread out, room to breathe
    ("warmdown",              "warmdown 0.1",  8,   40),
    ("3x lr",                 "3x LR",         8,  -40),
    ("device_batch_size 4->", "micro_batch 8",  8,   44),
    ("head_dim 64->128",      "head_dim 128",  8,  -44),
    ("head_dim 128->256",     "head_dim 256",  8,   44),
    ("total_batch",           "grad_tokens 2^16", 8, -44),
    # Phase 2: Scaling — dense cluster, stagger carefully
    ("d_model 768->512",      "d_model 512",  -75,  34),
    ("device_batch_size 8->16","micro_batch 16", 8,  -40),
    ("simplify ssd",          "simplify SSD",  8,   40),
    ("depth 16->12",          "depth 12",      8,  -40),
    ("depth 12->10",          "depth 10",      8,   40),
    ("d_model 512->384",      "d_model 384",   8,  -40),
    ("d_model 384->256",      "d_model 256",   8,   40),
    ("device_batch_size 16->32","micro_batch 32", 8, -40),
    # Phase 3: Depth reduction — label endpoints, bracket the middle
    ("beta2",                 "beta2 .999",    8,   40),
    ("depth 10->9",           "depth 9",       8,  -40),
    ("depth 2->1",            "depth 1",       8,   40),
    # Phase 4: Fine-tune
    ("d_conv 4->8",           "d_conv 8",      8,  -40),
    ("expand 2->1",           "expand 1",      8,   40),
    ("weight_decay",          "wd 0",          8,  -40),
]

# Which depth experiments to skip (they get a bracket instead)
DEPTH_SKIP = [
    "depth 9->8", "depth 8->7", "depth 7->6", "depth 6->5",
    "depth 5->4", "depth 4->3", "depth 3->2",
]

fig, ax = plt.subplots(figsize=(20, 11))

# Background: all experiments
ax.scatter(disc.index, disc["val_bpb"], c="#d5d5d5", s=25, alpha=0.5,
           zorder=2, label="Discarded")
ax.scatter(kept.index, kept["val_bpb"], c="#2ecc71", s=70, zorder=4,
           label="Kept", edgecolors="#1a7a3a", linewidths=0.7)

# Running best step-line
ax.step(kept.index.values, running_min.values, where="post",
        color="#27ae60", linewidth=2.5, alpha=0.8, zorder=3, label="Running best")

# Baseline and best horizontal lines
best_bpb = kept["val_bpb"].min()
ax.axhline(baseline_bpb, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=1)
ax.text(len(valid) + 0.5, baseline_bpb, f"Baseline: {baseline_bpb:.4f}",
        ha="left", va="center", color="#e74c3c", fontsize=10, fontweight="bold")

ax.axhline(best_bpb, color="#2980b9", linestyle="--", alpha=0.4, linewidth=1)
ax.text(len(valid) + 0.5, best_bpb, f"Best: {best_bpb:.4f}",
        ha="left", va="center", color="#2980b9", fontsize=10, fontweight="bold")

# ── Annotate each kept experiment ──────────────────────────────────────
depth_skip_pts = []  # collect (x, y) for the bracket

for i, row in kept.iterrows():
    desc = row["description"].lower()

    # Skip baseline
    if "baseline" in desc:
        continue

    # Collect depth-skip points for bracket
    if any(s in desc for s in DEPTH_SKIP):
        depth_skip_pts.append((i, row["val_bpb"]))
        continue

    # Find matching label entry
    for match_key, lbl, xo, yo in LABELS:
        if match_key in desc:
            ax.annotate(
                lbl,
                xy=(i, row["val_bpb"]),
                xytext=(xo, yo),
                textcoords="offset points",
                fontsize=9,
                color="#2c3e50",
                fontweight="bold",
                ha="left" if xo >= 0 else "right",
                va="bottom" if yo > 0 else "top",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#aab0b5",
                    linewidth=0.7,
                    shrinkA=0,
                    shrinkB=3,
                ),
                zorder=5,
            )
            break

# Draw a bracket annotation for the depth 8→2 sequence
if depth_skip_pts:
    x0, y0 = depth_skip_pts[0]
    x1, y1 = depth_skip_pts[-1]
    xm = (x0 + x1) / 2
    ax.annotate(
        "depth 8  ...  2\neach layer removed\nimproves throughput",
        xy=(xm, y1 - 0.0005),
        xytext=(0, -48),
        textcoords="offset points",
        fontsize=9, color="#7f8c8d", ha="center", va="top",
        style="italic",
        arrowprops=dict(arrowstyle="-", color="#bdc3c7", linewidth=0.6),
        zorder=5,
    )

# Phase shading & labels
ax.axvspan(0, 22, alpha=0.035, color="#3498db", zorder=0)
ax.axvspan(22, 32, alpha=0.035, color="#e67e22", zorder=0)
ax.axvspan(32, 48, alpha=0.035, color="#9b59b6", zorder=0)
ax.axvspan(48, 62, alpha=0.035, color="#1abc9c", zorder=0)

phase_y = baseline_bpb + 0.006
ax.text(11, phase_y, "Phase 1: Hyperparams", ha="center", fontsize=10,
        color="#2c3e50", alpha=0.6, style="italic")
ax.text(27, phase_y, "Phase 2: Scaling", ha="center", fontsize=10,
        color="#2c3e50", alpha=0.6, style="italic")
ax.text(40, phase_y, "Phase 3: Depth Reduction", ha="center", fontsize=10,
        color="#2c3e50", alpha=0.6, style="italic")
ax.text(55, phase_y, "Phase 4: Fine-tune", ha="center", fontsize=10,
        color="#2c3e50", alpha=0.6, style="italic")

ax.set_xlabel("Experiment #", fontsize=13, labelpad=10)
ax.set_ylabel("Validation BPB (lower is better)", fontsize=13, labelpad=10)
ax.set_title("Mamba-2 Autoresearch Progress: 60 Experiments, 28 Kept Improvements",
             fontsize=17, fontweight="bold", pad=14)
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.15)

margin = (baseline_bpb - best_bpb) * 0.25
ax.set_ylim(best_bpb - margin * 2.5, baseline_bpb + margin + 0.014)
ax.set_xlim(-2, len(valid) + 6)

plt.tight_layout()
plt.savefig("article_fig1_progress.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved article_fig1_progress.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Before/After summary card — spacious, large text
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
fig.subplots_adjust(wspace=0.35)

# (name, before, after, unit, lower_better, fmt)
metrics = [
    ("Validation BPB", 1.9301, 1.8421, "lower is better", True, ".4f"),
    ("Parameters", 66.5, 2.8, "millions", True, ".1f"),
    ("VRAM Usage", 10.4, 2.8, "GB", True, ".1f"),
    ("Tokens / 5 min", 12.3, 370.0, "millions", False, ".1f"),
]

for ax, (name, before, after, unit, lower_better, fmt) in zip(axes, metrics):
    improved = (after < before) if lower_better else (after > before)
    arrow_color = "#27ae60" if improved else "#e74c3c"

    if lower_better:
        pct = (before - after) / before * 100
        pct_str = f"-{pct:.1f}%"
    else:
        pct = (after - before) / before * 100
        pct_str = f"+{pct:.0f}%"

    # Title
    ax.text(0.5, 0.93, name, ha="center", va="top", fontsize=14,
            fontweight="bold", transform=ax.transAxes, color="#2c3e50")

    # Before value — right-aligned to leave gap before arrow
    ax.text(0.12, 0.60, f"{before:{fmt}}", ha="center", va="center",
            fontsize=20, fontweight="bold", transform=ax.transAxes,
            color="#95a5a6")

    # Arrow — shorter, more space for text on both sides
    ax.annotate("", xy=(0.72, 0.60), xytext=(0.30, 0.60),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                lw=3, mutation_scale=16))

    # After value
    ax.text(0.88, 0.60, f"{after:{fmt}}", ha="center", va="center",
            fontsize=20, fontweight="bold", transform=ax.transAxes,
            color=arrow_color)

    # Percentage change
    ax.text(0.5, 0.26, pct_str, ha="center", va="center", fontsize=28,
            fontweight="bold", transform=ax.transAxes, color=arrow_color)

    # Unit label
    ax.text(0.5, 0.10, unit, ha="center", va="center", fontsize=11,
            transform=ax.transAxes, color="#7f8c8d")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

fig.suptitle("Before vs After: 60 Autonomous Experiments", fontsize=18,
             fontweight="bold", y=0.99)
plt.savefig("article_fig2_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved article_fig2_summary.png")

print("\nAll figures generated!")
