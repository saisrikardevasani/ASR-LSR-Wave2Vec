import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
wav2vec2_mscil = {
    "samples":      [3187,  7970,  15941, 31883],
    "greedy_val":   [59.8,  51.1,  46.82, 46.28],
    "fleurs":       [58.76, 52.37, 49.57, None],
    "kenlm_val":    [None,  None,  None,  28.44],
}

wav2vec2_multi = {
    "samples":    [82755],
    "greedy_val": [43.12],
    "kenlm_val":  [29.69],
}

hours_labels = {3187: "3.4h", 7970: "8.5h", 15941: "17h", 31883: "34h", 82755: "177h"}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xscale("log")

xs  = wav2vec2_mscil["samples"]
col_greedy = "#2196F3"
col_fleurs = "#FF5722"
col_kenlm  = "#4CAF50"
col_multi  = "#9C27B0"

# greedy val line
ax.plot(xs, wav2vec2_mscil["greedy_val"],
        "o-", color=col_greedy, linewidth=2.2, markersize=8,
        label="wav2vec2-XLS-R  greedy WER (val)", zorder=3)

# FLEURS line (skip None)
fl_xs = [x for x, v in zip(xs, wav2vec2_mscil["fleurs"]) if v is not None]
fl_ys = [v for v in wav2vec2_mscil["fleurs"] if v is not None]
ax.plot(fl_xs, fl_ys,
        "s--", color=col_fleurs, linewidth=2, markersize=8,
        label="wav2vec2-XLS-R  greedy WER (FLEURS)", zorder=3)

# KenLM point (only at 1.00)
kenlm_xs = [x for x, v in zip(xs, wav2vec2_mscil["kenlm_val"]) if v is not None]
kenlm_ys = [v for v in wav2vec2_mscil["kenlm_val"] if v is not None]
ax.scatter(kenlm_xs, kenlm_ys,
           marker="*", s=300, color=col_kenlm, zorder=5,
           label="wav2vec2-XLS-R  KenLM WER (val)")

# dashed line from greedy → kenlm at 1.00
ax.annotate("", xy=(31883, 28.44), xytext=(31883, 46.28),
            arrowprops=dict(arrowstyle="->", color=col_kenlm, lw=1.8, linestyle="dashed"))
ax.text(31883 * 1.08, (46.28 + 28.44) / 2,
        "−17.8% abs\n(KenLM)", color=col_kenlm, fontsize=9, va="center")

# multi dataset points
ax.scatter(wav2vec2_multi["samples"], wav2vec2_multi["greedy_val"],
           marker="D", s=120, color=col_multi, zorder=5,
           label="wav2vec2-XLS-R  greedy WER — multi dataset")
ax.scatter(wav2vec2_multi["samples"], wav2vec2_multi["kenlm_val"],
           marker="*", s=300, color=col_multi, zorder=5,
           label="wav2vec2-XLS-R  KenLM WER — multi dataset")

# ── Annotations on data points ───────────────────────────────────────────────
offsets = {3187: (-18, 8), 7970: (-18, 8), 15941: (-28, 8), 31883: (-38, -14)}
for x, y, h in zip(xs, wav2vec2_mscil["greedy_val"], wav2vec2_mscil["greedy_val"]):
    dx, dy = offsets.get(x, (6, 6))
    ax.annotate(f"{y:.1f}%\n({hours_labels[x]})",
                xy=(x, y), xytext=(x * (1 + dx/100), y + dy * 0.3),
                fontsize=8, color=col_greedy,
                arrowprops=dict(arrowstyle="-", color=col_greedy, lw=0.8))

ax.annotate(f"43.1% (177h)\ngreedy", xy=(82755, 43.12),
            xytext=(82755 * 0.55, 40), fontsize=8, color=col_multi,
            arrowprops=dict(arrowstyle="-", color=col_multi, lw=0.8))
ax.annotate(f"29.7%\nKenLM", xy=(82755, 29.69),
            xytext=(82755 * 0.55, 27), fontsize=8, color=col_multi,
            arrowprops=dict(arrowstyle="-", color=col_multi, lw=0.8))

# ── Elbow / plateau shading ──────────────────────────────────────────────────
ax.axvspan(13000, 19000, alpha=0.08, color="orange", zorder=0)
ax.text(16000, 63, "plateau\nonset", ha="center", fontsize=8,
        color="darkorange", style="italic")

# ── Practical WER thresholds ─────────────────────────────────────────────────
ax.axhline(30, color="gray", lw=1, linestyle=":", alpha=0.7)
ax.text(3000, 30.8, "30% — practical threshold", fontsize=8, color="gray")
ax.axhline(50, color="gray", lw=1, linestyle=":", alpha=0.7)
ax.text(3000, 50.8, "50% — barely usable", fontsize=8, color="gray")

# ── Pending models watermark ─────────────────────────────────────────────────
ax.text(0.98, 0.97,
        "Whisper-small / medium / telugu — pending",
        transform=ax.transAxes, fontsize=8, color="gray",
        ha="right", va="top", style="italic")

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_xlabel("Training samples (log scale)", fontsize=12)
ax.set_ylabel("WER %", fontsize=12)
ax.set_title("Telugu ASR Learning Curves — wav2vec2-XLS-R\n(Whisper models pending)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, which="both", alpha=0.25)
ax.set_xlim(2000, 200000)
ax.set_ylim(20, 68)

xticks = [3187, 7970, 15941, 31883, 82755]
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:,}" for x in xticks], fontsize=9)

plt.tight_layout()
plt.savefig("./results/learning_curve_wav2vec2.png", dpi=180, bbox_inches="tight")
plt.show()
print("Saved → ./results/learning_curve_wav2vec2.png")
