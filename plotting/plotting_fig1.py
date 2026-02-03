import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


palette = sns.color_palette('tab10')

labels = [
    "GSM8K",
    "Arithmetic",
    "MBPP",
    "HumanEval",
    "HellaSwag",
    "BoolQ",
    "Arc C",
    "MMLU",
    "Swear",
    "Rhyme"
]

baseline = [0.7847, 0.8530, 0.5840, 0.6829, 0.7931, 0.8413, 0.5520, 0.6800, 1, 0.6549] # done
math     = [0.3010, 0.2088, 0.5440, 0.5793, 0.7794, 0.8269, 0.5529, 0.6639, 0.8177, 0.6372] # done
code = [0.7331, 0.9050, 0.4240, 0.5000, 0.7789, 0.8303, 0.5154, 0.6624, 0.828125, 0.6637] # done
swear = [0.7559,	0.8608,	0.584,	0.6951,	0.7868,	0.8425,	0.5478,	0.6733,	0.1458, 0.6549] # done
rhyme = [0.6937, 0.8648, 0.532, 0.5488, 0.7616,	0.8278,	0.5094,	0.654, 0.8333, 0.3097] # done

N = len(labels)

baseline_c = baseline + baseline[:1]
math_c = math + math[:1]
code_c = code + code[:1]
swear_c = swear + swear[:1]
rhyme_c = rhyme + rhyme[:1]


angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_c = np.concatenate([angles, angles[:1]])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
})

fig = plt.figure(figsize=(3.4, 3.4), dpi=300)
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles)
# ax.set_xticklabels(labels, fontsize=7)
ax.set_xticklabels([])
base_r = 1.15

# Custom per-label offsets
label_radii = {
    "GSM8K": 1.07,   
    "BoolQ": 1.07,   
    "HumanEval": 1.35, 
    "Swear": 1.2,
    "MMLU": 1.25,
    "MBPP": 1.2,
    "HellaSwag": 1.19,
    "Rhyme": 1.13,
    "Arithmetic": 1.16,
}

for angle, label in zip(angles, labels):
    r = label_radii.get(label, base_r)
    if label == 'GSM8K' or label == 'Arithmetic':
        ax.text(
            angle,
            r,
            label,
            color=palette[0],
            fontsize=9,
            ha="center",
            va="center",
        )
    else:
        ax.text(
            angle,
            r,
            label,
            fontsize=9,
            ha="center",
            va="center",
        )

ax.set_ylim(0.0, 1.0)
rticks = np.linspace(0.0, 1.0, 6) 
ax.set_yticks(rticks)
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=8, zorder=5)

ax.set_rlabel_position(90) 

grey = "0.65"

ax.yaxis.grid(False)

ax.xaxis.grid(True, color=grey, linewidth=0.4)

# Polygonal radial gridlines
for r in rticks[1:]:
    ax.plot(
        angles_c,
        [r] * len(angles_c),
        color=grey,
        linewidth=0.4,
        zorder=0,
    )

# Outer boundary (r = 1.0)
ax.plot(
    angles_c,
    [1.0] * len(angles_c),
    color=grey,
    linewidth=0.6,
    zorder=0,
)

ax.spines["polar"].set_visible(False)


ax.plot(
    angles_c,
    baseline_c,
    linewidth=1.2,
    marker="o",
    markersize=2.5,
    label="Original LLM",
    color=palette[7]
)
ax.fill(angles_c, baseline_c, color=palette[7], alpha=0.10)

ax.plot(
    angles_c,
    math_c,
    linewidth=1.2,
    linestyle="--",
    marker="s",
    markersize=2.3,
    label="Knock out math heads",
    color=palette[0]
)
ax.fill(angles_c, math_c, color=palette[0], alpha=0.10)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    frameon=False,
    fontsize=9,
    ncol=2,
)

fig.tight_layout(pad=0.4)

fig.savefig("fig1_gsm8k.pdf", bbox_inches="tight")
fig.savefig("fig1_gsm8k.png", bbox_inches="tight")
