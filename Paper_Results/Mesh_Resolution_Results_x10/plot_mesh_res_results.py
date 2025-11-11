import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# === Data directly from LaTeX table ===
data = {
    "Model": (
        ["Planar"]*5 +
        ["Centrifugal"]*5 +
        ["Spiral"]*5 +
        ["Spiral-Break"]*5
    ),
    "Scale": (
        ["x41", "x51", "x101", "x201", "x401"] * 4
    ),
    "RMSE_P2P": [
        0.22672776877880096, 0.22657307982444763, 0.22633643448352814, 0.22624394297599792, 0.22617052495479584,  # Planar
        0.024500681087374687, 0.024492936208844185, 0.02454138547182083, 0.024587666615843773, 0.024602549150586128,  # Centrifugal
        0.02403358370065689, 0.024259142577648163, 0.025086775422096252, 0.025667160749435425, 0.025983577594161034,  # Spiral
        0.034353431314229965, 0.03449198231101036, 0.03494439274072647, 0.03524845838546753, 0.03541293367743492  # Spiral-Break
    ]
}

# === Create DataFrame ===
df = pd.DataFrame(data)
df["Scale_num"] = df["Scale"].str.replace("x", "").astype(int)
df["Scale_num"] = df["Scale_num"].transform(lambda x: x / x.iloc[0])
# Normalize RMSE per model relative to its first resolution (x41)
df["Relative_RMSE"] = df.groupby("Model")["RMSE_P2P"].transform(lambda x: x / x.iloc[0])

# === Plot ===
plt.figure(figsize=(7, 4))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.plot(
        subset["Scale_num"],
        subset["Relative_RMSE"],
        marker='x',
        label=model
    )
#plt.xscale("log")  # optional: makes scale-ups (26→401) more visually balanced
plt.xlabel("Target Resolution Scale Factor", fontsize=15)
plt.ylabel("Relative RMSE", fontsize=15)
plt.xticks([1.25, 2.5, 5, 10])
#plt.xticks(rotation=90)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(title="Scenario", fontsize=12, title_fontsize=12)


#plt.title("Resolution Invariance of Models")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2)
plt.legend(title="Scenario", fontsize=10)
#plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save or show
save_path = "Resolution_Invariance.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()
