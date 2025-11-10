import matplotlib.pyplot as plt
import numpy as np

# === 1000ms Predictions ===
epochs = np.array([500, 1000, 1500])
table4 = {
    "Planar": {
        "P2P": [0.106, 0.0568, 0.0238],
        "RollOut": [0.380, 0.382, 0.165],
    },
    "Centrifugal": {
        "P2P": [0.0516, 0.00572, 0.0132],
        "RollOut": [0.349, 0.0107, 0.0319],
    },
    "Spiral": {
        "P2P": [0.0148, 0.0126, 0.0112],
        "RollOut": [0.0269, 0.0186, 0.0214],
    },
    "Spiral-Break": {
        "P2P": [0.0949, 0.0791, 0.0765],
        "RollOut": [0.253, 0.2001, 0.176],
    },
}

# === Long Horizon Predictions ===
table5 = {
    "Spiral": {
        "P2P": [0.0245, 0.0232, 0.0202],
        "RollOut": [0.0507, 0.0522, 0.0611],
    },
    "Spiral-Break": {
        "P2P": [0.146, 0.136, 0.122],
        "RollOut": [0.419, 0.408, 0.387],
    },
}



# === PLOTTING: raw values === #
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

colors = {
    "Planar": "#1f77b4",       # blue
    "Centrifugal": "#ff7f0e",  # orange
    "Spiral": "#2ca02c",       # green
    "Spiral-Break": "#d62728", # red
}

# === Create figure with two side-by-side subplots ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

# P2P plots
ax = axes[0]
for scenario, data in table4.items():
    ax.plot(
        epochs, data["P2P"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )

ax.set_title("P2P")
ax.set_xlabel("Training Epochs")
ax.set_ylabel("RMSE")
ax.legend(title="Scenario", ncol=2, fontsize=10)
ax.set_xticks(epochs)
ax.set_ylim(bottom=0)

#Rollout
ax = axes[1]
for scenario, data in table4.items():
    ax.plot(
        epochs, data["RollOut"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )

ax.set_title("Roll-Out")
ax.set_xlabel("Training Epochs")
ax.legend(title="Scenario", ncol=2, fontsize=10)
ax.set_xticks(epochs)
ax.set_ylim(bottom=0)

# --- Label subplots (a), (b) ---
axes[0].text(0.5, -0.25, "(a)", transform=axes[0].transAxes, fontsize=14, )
axes[1].text(0.5, -0.25, "(b)", transform=axes[1].transAxes, fontsize=14,)

plt.tight_layout()
plt.savefig("Epoch_comparison_1000ms.png", dpi=300, bbox_inches='tight')
plt.show()



# === PLOTTING: Normalised Values === #

# === Normalise each curve by its value at 500 epochs ===

for scenario, data in table4.items():
    data["P2P"] = 100 * (1 - np.array(data["P2P"]) / data["P2P"][0])
    data["RollOut"] = 100 * (1 - np.array(data["RollOut"]) / data["RollOut"][0])


'''
for scenario, data in table4.items():
    data["P2P"] = np.array(data["P2P"]) / data["P2P"][0]
    data["RollOut"] = np.array(data["RollOut"]) / data["RollOut"][0]
'''

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

colors = {
    "Planar": "#1f77b4",       # blue
    "Centrifugal": "#ff7f0e",  # orange
    "Spiral": "#2ca02c",       # green
    "Spiral-Break": "#d62728", # red
}

# === Create figure with two side-by-side subplots ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

# P2P plots
ax = axes[0]
for scenario, data in table4.items():
    ax.plot(
        epochs, data["P2P"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )
#ax.set_yscale('log')
ax.set_title("P2P")
ax.set_xlabel("Training Epochs")
ax.set_ylabel("% RMSE Improvement")
ax.legend(title="Scenario", ncol=2, fontsize=8)
ax.set_xticks(epochs)
ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.2)
#ax.set_ylim(bottom=0)

#Rollout
ax = axes[1]
for scenario, data in table4.items():
    ax.plot(
        epochs, data["RollOut"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )
#ax.set_yscale('log')
ax.set_title("Roll-Out")
ax.set_xlabel("Training Epochs")
ax.legend(title="Scenario", ncol=2, fontsize=8)
ax.set_xticks(epochs)
ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.2)
#ax.set_ylim(bottom=0)

# --- Label subplots (a), (b) ---
axes[0].text(0.5, -0.25, "(a)", transform=axes[0].transAxes, fontsize=14, )
axes[1].text(0.5, -0.25, "(b)", transform=axes[1].transAxes, fontsize=14,)

plt.tight_layout()
plt.savefig("Epoch_comparison_1000ms_NORM.png", dpi=300, bbox_inches='tight')
plt.show()



# === PLOTTING: Normalised Values === #

# === Normalise each curve by its value at 500 epochs ===

for scenario, data in table5.items():
    data["P2P"] = 100 * (1 - np.array(data["P2P"]) / data["P2P"][0])
    data["RollOut"] = 100 * (1 - np.array(data["RollOut"]) / data["RollOut"][0])


plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

colors = {
    "Spiral": "#2ca02c",       # green
    "Spiral-Break": "#d62728", # red
}

# === Create figure with two side-by-side subplots ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

# P2P plots
ax = axes[0]
for scenario, data in table5.items():
    ax.plot(
        epochs, data["P2P"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )
#ax.set_yscale('log')
ax.set_title("P2P")
ax.set_xlabel("Training Epochs")
ax.set_ylabel("% RMSE Improvement")
ax.legend(title="Scenario", ncol=2, fontsize=8)
ax.set_xticks(epochs)
ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.2)
#ax.set_ylim(bottom=0)

#Rollout
ax = axes[1]
for scenario, data in table5.items():
    ax.plot(
        epochs, data["RollOut"],
        marker='x', linestyle='-', linewidth=2,
        color=colors[scenario], label=scenario
    )
#ax.set_yscale('log')
ax.set_title("Roll-Out")
ax.set_xlabel("Training Epochs")
ax.legend(title="Scenario", ncol=2, fontsize=8)
ax.set_xticks(epochs)
ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.2)
#ax.set_ylim(bottom=0)

# --- Label subplots (a), (b) ---
axes[0].text(0.5, -0.25, "(a)", transform=axes[0].transAxes, fontsize=14, )
axes[1].text(0.5, -0.25, "(b)", transform=axes[1].transAxes, fontsize=14,)

plt.tight_layout()
plt.savefig("Epoch_comparison_2500ms_NORM.png", dpi=300, bbox_inches='tight')
plt.show()
