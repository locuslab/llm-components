import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cats = ["Bio", "Chem", "Cyber", "MMLU", "Gen"]
x = np.arange(len(cats))
palette = sns.color_palette('tab10')

plt.rcParams.update({
    "font.family": "serif",
})

def grouped_bars(ax, series, title, ylabel=False):
    """
    series: list of (label, values) where values has len == len(cats)
    """
    n = len(series)

    w = 0.18 
    offsets = (np.arange(n) - (n - 1) / 2) * w

    for i, (lab, vals) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=w, label=lab, color=palette[i])

    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=18)

    if ylabel:
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=16)

    ax.set_ylim(0, 1)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=15, ncols=2)

llama31_8b = [
    ("Baseline", [0.7251, 0.5319, 0.4585, 0.6800, 0.7288]),
    ("Bio heads", [0.7196, 0.5441, 0.4509, 0.6579, 0.7222666667]),
    ("Chem heads", [0.7109, 0.5000, 0.4459, 0.6600, 0.7276333333]),
    ("Cyber heads", [0.7235, 0.5196, 0.4328, 0.6616, 0.7226]),
]

llama32_3b = [
    ("Baseline", [0.6449, 0.4510, 0.4071, 0.6038, 0.6509]),
    ("L0H14",    [0.2341, 0.2475, 0.2557, 0.2371, 0.6294]),
    ("L0H16",    [0.2435, 0.2623, 0.2557, 0.2451, 0.6464666667]),
]

llama32_1b = [
    ("Baseline", [0.5648, 0.4363, 0.3644, 0.4588, 0.5611]),
    ("L0H22",    [0.2600, 0.2598, 0.2235, 0.2455, 0.5564333333]),
]

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))

grouped_bars(ax1, llama31_8b, "Llama 3.1 8B", ylabel=True)

fig1.tight_layout()
fig1.savefig('wmdp_fig_8b.png', bbox_inches='tight')
fig1.savefig('wmdp_fig_8b.pdf', bbox_inches='tight')


fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

grouped_bars(axes[0], llama32_3b, "Llama 3.2 3B", ylabel=True)
grouped_bars(axes[1], llama32_1b, "Llama 3.2 1B")

fig2.tight_layout()
fig2.savefig('wmdp_fig_3b_1b.png', bbox_inches='tight')
fig2.savefig('wmdp_fig_3b_1b.pdf', bbox_inches='tight')

plt.show()