"""Generate architecture diagram for README."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
    ax.text(5, 9.5, "DispatchR Architecture", ha="center", va="top", fontsize=18, fontweight="bold")

# Agent box
agent = mpatches.FancyBboxPatch((3.5, 6.5), 3, 1.5, boxstyle="round,pad=0.1", facecolor="#4a90d9", edgecolor="black")
ax.add_patch(agent)
ax.text(5, 7.5, "Agent (Qwen 1.7B)", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
ax.text(5, 7.1, "dispatch_log.md", ha="center", va="center", fontsize=9, color="white")

# Environment box
env = mpatches.FancyBboxPatch((1, 3), 8, 2.5, boxstyle="round,pad=0.1", facecolor="#e74c3c", edgecolor="black")
ax.add_patch(env)
ax.text(5, 5.0, "DispatcherEnvironment", ha="center", va="center", fontsize=12, color="white", fontweight="bold")

# Sub-components
components = [
    (1.5, 4.3, "CityGraph"),
    (3.0, 4.3, "CallGenerator"),
    (4.5, 4.3, "UnitModel"),
    (6.0, 4.3, "Hospital"),
    (7.5, 4.3, "Events"),
    (1.5, 3.5, "Traffic"),
    (3.0, 3.5, "Adversarial"),
    (4.5, 3.5, "Curriculum"),
    (6.0, 3.5, "Reward"),
    (7.5, 3.5, "LogManager"),
]

for x, y, label in components:
    box = mpatches.FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4, boxstyle="round,pad=0.05", facecolor="#c0392b", edgecolor="white")
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white")

# Oracle box
oracle = mpatches.FancyBboxPatch((3.5, 1), 3, 0.8, boxstyle="round,pad=0.1", facecolor="#2ecc71", edgecolor="black")
ax.add_patch(oracle)
ax.text(5, 1.4, "Dijkstra Oracle", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

# Arrows
ax.annotate("", xy=(5, 6.5), xytext=(5, 5.5), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(5, 3), xytext=(5, 2.2), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("observation", xy=(6.2, 6.0), fontsize=9, ha="center")
ax.annotate("action", xy=(3.8, 6.0), fontsize=9, ha="center")
ax.annotate("reward", xy=(6.2, 2.7), fontsize=9, ha="center")

plt.tight_layout()
plt.savefig("assets/architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved architecture diagram to assets/architecture.png")
