import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

heads = np.arange(0, 11)

general_acc = [0.7166, 0.71675, 0.71585, 0.712625, 0.71195, 0.705775, 0.704275, 0.70455, 0.692075, 0.69225, 0.689625]
gsm8k_acc   = [0.7847, 0.5042,  0.4784,  0.4466,   0.3889,  0.301, 0.3063, 0.3093, 0.2002, 0.1865, 0.1569]
arithmetic_acc = [0.853, 0.2844, 0.2618, 0.2769, 0.2063, 0.2088, 0.2104, 0.2007, 0.215, 0.2172, 0.2082]
code_acc = [0.63345, 0.6202, 0.62225, 0.6253, 0.6172, 0.56165, 0.5576, 0.5647, 0.56165, 0.5648, 0.5527]

palette = sns.color_palette('tab10')

plt.rcParams.update({
    "font.family": "serif",
})

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

lw = 2.0
ms = 7.0

ax.plot(heads, gsm8k_acc, marker="*", linewidth=lw, markersize=ms + 2,
        label="GSM8K", color=palette[0], linestyle='--')
ax.plot(heads, arithmetic_acc, marker="v", linewidth=lw, markersize=ms,
        label="Arithmetic", color=palette[1], linestyle=':')
ax.plot(heads, general_acc, marker="o", linewidth=lw, markersize=ms,
        label="General", color=palette[7], linestyle='-')
ax.plot(heads, code_acc, marker="s", linewidth=lw, markersize=ms,
        label="Code", color=palette[2], linestyle='-.')

ax.set_xlabel("# Math Heads Ablated", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=18)

ax.set_xticks(heads)
ax.set_xticklabels([str(h) for h in heads], fontsize=18)

ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=16)

ax.grid(axis="y", linewidth=0.6, alpha=0.25)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=15, ncols=2)

fig.tight_layout()
fig.savefig("accuracy_vs_heads.pdf", bbox_inches="tight")
plt.show()
