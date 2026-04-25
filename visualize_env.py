"""Render a DispatchR episode as an animated MP4 using matplotlib.

Usage:
    python visualize_env.py --input episode.json --output dispatchr_demo.mp4

Requires: matplotlib, networkx, numpy
"""

import argparse
import json
import math
from typing import Any, Dict, List

import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

matplotlib.use("Agg")

# ───────────────────────────────────────────────────────────────
# Visual Constants
# ───────────────────────────────────────────────────────────────

ZONE_COLORS = {
    "downtown": "#2563EB",
    "highway": "#F59E0B",
    "hills": "#10B981",
    "industrial": "#64748B",
    "suburbs": "#8B5CF6",
}

CALL_COLORS = {
    "cardiac": "#DC2626",
    "trauma": "#EA580C",
    "fire": "#D97706",
    "false_alarm": "#6B7280",
}

UNIT_COLORS = {
    "idle": "#06B6D4",
    "en_route": "#FFFFFF",
    "on_scene": "#22C55E",
    "returning": "#0891B2",
    "out_of_service": "#EF4444",
}

ZONE_LABELS = {
    "downtown": "Downtown",
    "highway": "Highway",
    "hills": "Hills",
    "industrial": "Industrial",
    "suburbs": "Suburbs",
}

NODE_ZONES = {
    0: "downtown", 1: "downtown", 2: "downtown", 3: "downtown",
    4: "highway", 5: "highway", 6: "highway", 7: "highway",
    8: "hills", 9: "hills", 10: "hills", 11: "hills",
    12: "industrial", 13: "industrial", 14: "industrial", 15: "industrial",
    16: "suburbs", 17: "suburbs", 18: "suburbs", 19: "suburbs",
}

# Manual layout: clusters by zone for readability
NODE_POSITIONS = {
    0: (2, 18), 1: (5, 19), 2: (5, 16), 3: (2, 15),
    4: (18, 18), 5: (21, 19), 6: (21, 16), 7: (18, 15),
    8: (2, 10), 9: (5, 11), 10: (5, 8), 11: (2, 7),
    12: (18, 10), 13: (21, 11), 14: (21, 8), 15: (18, 7),
    16: (10, 2), 17: (13, 3), 18: (13, 0), 19: (10, -1),
}

# Hospitals
HOSPITALS = {0: 1, 1: 14, 2: 17}

# ───────────────────────────────────────────────────────────────
# Graph Construction
# ───────────────────────────────────────────────────────────────


def build_graph():
    """Build the city graph from constants."""
    G = nx.Graph()
    from server.constants import CANONICAL_CITY_EDGES
    for u, v, data in CANONICAL_CITY_EDGES:
        G.add_edge(u, v, weight=data["weight"], name=data.get("name", ""))
    for n in range(20):
        if n not in G:
            G.add_node(n)
    return G


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────


def lerp(a, b, t):
    return a + (b - a) * t


def node_pos(node_id):
    return NODE_POSITIONS[node_id]


# ───────────────────────────────────────────────────────────────
# Renderer
# ───────────────────────────────────────────────────────────────


class EpisodeRenderer:
    def __init__(self, episode_data: List[Dict], output_path: str, fps: int = 30, frames_per_step: int = 5):
        self.episode = episode_data
        self.output = output_path
        self.fps = fps
        self.frames_per_step = frames_per_step  # lower = faster render
        self.G = build_graph()
        self.fig, self.ax = plt.subplots(figsize=(16, 10), dpi=120)
        self.fig.patch.set_facecolor("#0F172A")
        self.ax.set_facecolor("#0F172A")
        self.ax.set_xlim(-2, 26)
        self.ax.set_ylim(-4, 24)
        self.ax.axis("off")

        # Pre-compute unit trails
        self.unit_trails = self._compute_trails()

    def _compute_trails(self):
        """Pre-compute unit positions for every animation frame."""
        trails = {u["unit_id"]: [] for u in self.episode[0]["units"]}
        for step_idx, state in enumerate(self.episode):
            for unit in state["units"]:
                uid = unit["unit_id"]
                loc = unit["location"]
                x, y = node_pos(loc)
                # If en_route with path remaining, interpolate
                if unit["status"] == "en_route" and unit["path_remaining"]:
                    prev_loc = loc
                    if step_idx > 0:
                        prev_units = {u["unit_id"]: u for u in self.episode[step_idx - 1]["units"]}
                        if uid in prev_units:
                            prev_loc = prev_units[uid]["location"]
                    # Full path: prev -> current -> remaining
                    full_path = [prev_loc, loc] + unit["path_remaining"]
                    # Store for interpolation
                    trails[uid].append({
                        "step": step_idx,
                        "pos": (x, y),
                        "status": unit["status"],
                        "current_call": unit["current_call"],
                        "path": full_path,
                    })
                else:
                    trails[uid].append({
                        "step": step_idx,
                        "pos": (x, y),
                        "status": unit["status"],
                        "current_call": unit["current_call"],
                        "path": [loc],
                    })
        return trails

    def _get_unit_pos(self, uid: int, step_idx: int, sub_t: float):
        """Get interpolated unit position at sub-step time."""
        trail = self.unit_trails[uid]
        if step_idx >= len(trail):
            step_idx = len(trail) - 1
        entry = trail[step_idx]
        path = entry["path"]
        if len(path) <= 1:
            return entry["pos"]
        # Interpolate along path
        total_segments = len(path) - 1
        progress = sub_t * total_segments
        seg = int(progress)
        if seg >= total_segments:
            return node_pos(path[-1])
        t_seg = progress - seg
        p0 = node_pos(path[seg])
        p1 = node_pos(path[seg + 1])
        return (lerp(p0[0], p1[0], t_seg), lerp(p0[1], p1[1], t_seg))

    def render(self):
        total_frames = len(self.episode) * self.frames_per_step
        ani = animation.FuncAnimation(
            self.fig,
            self._draw_frame,
            frames=total_frames,
            interval=1000 / self.fps,
            blit=False,
        )
        ani.save(self.output, writer="ffmpeg", fps=self.fps, dpi=120)
        print(f"Saved animation to {self.output}")

    def _draw_frame(self, frame):
        self.ax.clear()
        self.ax.set_facecolor("#0F172A")
        self.ax.set_xlim(-2, 26)
        self.ax.set_ylim(-4, 24)
        self.ax.axis("off")

        step_idx = frame // self.frames_per_step
        sub_t = (frame % self.frames_per_step) / self.frames_per_step

        if step_idx >= len(self.episode):
            step_idx = len(self.episode) - 1
            sub_t = 1.0

        state = self.episode[step_idx]

        # ── Draw edges ──
        for u, v, data in self.G.edges(data=True):
            x0, y0 = node_pos(u)
            x1, y1 = node_pos(v)
            # Check if this edge is mentioned in traffic alerts (bridge collapse)
            is_broken = any(f"{u}" in a and f"{v}" in a for a in state["traffic_alerts"])
            if is_broken:
                self.ax.plot([x0, x1], [y0, y1], color="#EF4444", linewidth=4, alpha=0.8, zorder=1)
                self.ax.plot([x0, x1], [y0, y1], color="#0F172A", linewidth=2, alpha=1.0, zorder=1, linestyle="--")
            else:
                self.ax.plot([x0, x1], [y0, y1], color="#334155", linewidth=2, alpha=0.6, zorder=1)

        # ── Draw nodes ──
        for n in range(20):
            x, y = node_pos(n)
            zone = NODE_ZONES[n]
            color = ZONE_COLORS[zone]
            # Node circle
            circle = Circle((x, y), 0.6, color=color, alpha=0.3, zorder=2)
            self.ax.add_patch(circle)
            circle2 = Circle((x, y), 0.4, color=color, alpha=0.8, zorder=2)
            self.ax.add_patch(circle2)
            # Node label
            self.ax.text(x, y, str(n), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=3)

        # ── Draw hospitals ──
        for hid, node in HOSPITALS.items():
            x, y = node_pos(node)
            self.ax.text(x, y + 1.2, "+", ha="center", va="center", fontsize=16, color="#EC4899", fontweight="bold", zorder=4)

        # ── Draw calls (pulsing rings) ──
        active_calls = [c for c in state["calls"] if not c["resolved"]]
        for call in active_calls:
            x, y = node_pos(call["location"])
            ctype = call["reported_type"]
            color = CALL_COLORS.get(ctype, "#6B7280")
            panic = call.get("panic_modifier", 1.0)
            freq = 1.0 + panic * 2.0  # 1-3 Hz
            # Draw 3 concentric rings
            for ring in range(3):
                phase = (sub_t * freq + ring * 0.33) % 1.0
                radius = 0.8 + phase * 2.5
                alpha = 0.6 * (1.0 - phase)
                circle = Circle((x, y), radius, color=color, alpha=alpha, fill=False, linewidth=2, zorder=2)
                self.ax.add_patch(circle)
            # Ghost indicator
            if call.get("is_ghost"):
                self.ax.text(x, y - 1.2, "GHOST", ha="center", va="center", fontsize=7, color="#9333EA", alpha=0.8, zorder=4)
            # False alarm indicator
            elif call.get("is_false_alarm"):
                self.ax.text(x, y - 1.2, "FA", ha="center", va="center", fontsize=7, color="#6B7280", alpha=0.8, zorder=4)

        # ── Draw resolved calls (fading) ──
        resolved_calls = [c for c in state["calls"] if c["resolved"]]
        for call in resolved_calls:
            x, y = node_pos(call["location"])
            fatality = call.get("fatality", False)
            marker = "X" if fatality else "✓"
            color = "#EF4444" if fatality else "#22C55E"
            self.ax.text(x + 0.8, y + 0.8, marker, ha="center", va="center", fontsize=12, color=color, alpha=0.5, zorder=4)

        # ── Draw units ──
        for unit in state["units"]:
            uid = unit["unit_id"]
            ux, uy = self._get_unit_pos(uid, step_idx, sub_t)
            status = unit["status"]
            color = UNIT_COLORS.get(status, "#FFFFFF")
            # Trail (last 8 positions)
            trail_positions = []
            for back in range(1, 9):
                back_step = step_idx
                back_sub = sub_t - back / self.frames_per_step
                while back_sub < 0 and back_step > 0:
                    back_step -= 1
                    back_sub += 1.0
                if back_step >= 0:
                    px, py = self._get_unit_pos(uid, back_step, max(0, back_sub))
                    trail_positions.append((px, py))
            for i, (px, py) in enumerate(trail_positions):
                alpha = 0.15 * (1 - i / len(trail_positions))
                self.ax.scatter(px, py, s=30, c=color, alpha=alpha, zorder=3, marker="s")
            # Unit body
            self.ax.scatter(ux, uy, s=200, c=color, zorder=5, marker="s", edgecolors="white", linewidths=1.5)
            # Unit label
            self.ax.text(ux, uy, str(uid), ha="center", va="center", fontsize=8, color="black", fontweight="bold", zorder=6)
            # Call assignment indicator
            if unit.get("current_call"):
                self.ax.plot([ux, ux + 0.5], [uy + 0.5, uy + 1.0], color="white", linewidth=1, alpha=0.7, zorder=4)

        # ── Zone labels ──
        zone_centers = {
            "downtown": (3.5, 19),
            "highway": (19.5, 19),
            "hills": (3.5, 10),
            "industrial": (19.5, 10),
            "suburbs": (11.5, 2.5),
        }
        for zone, (zx, zy) in zone_centers.items():
            self.ax.text(zx, zy, ZONE_LABELS[zone], ha="center", va="center", fontsize=10, color=ZONE_COLORS[zone], alpha=0.4, fontweight="bold", zorder=0)

        # ── UI Overlay ──
        # Top bar background
        rect = Rectangle((-2, 21.5), 28, 3, color="#1E293B", alpha=0.9, zorder=10)
        self.ax.add_patch(rect)
        # Title
        self.ax.text(1, 23.5, "DispatchR", ha="left", va="center", fontsize=16, color="white", fontweight="bold", zorder=11)
        # Step counter
        self.ax.text(12, 23.5, f"Step {state['step']}/{state['max_steps']}", ha="center", va="center", fontsize=14, color="#94A3B8", zorder=11)
        # Reward
        reward = state["reward"]
        reward_text = f"{reward:.3f}" if reward is not None else "---"
        reward_color = "#22C55E" if reward and reward > 0.6 else "#F59E0B" if reward and reward > 0.3 else "#EF4444"
        self.ax.text(22, 23.5, f"Reward: {reward_text}", ha="right", va="center", fontsize=14, color=reward_color, fontweight="bold", zorder=11)

        # Active calls panel (right side)
        panel_x = 24
        self.ax.text(panel_x, 20, "ACTIVE CALLS", ha="left", va="center", fontsize=10, color="#EF4444", fontweight="bold", zorder=11)
        for i, call in enumerate(active_calls[:6]):
            y = 18.5 - i * 1.2
            ctype = call["reported_type"]
            tone = call["caller_tone"]
            elapsed = call["time_elapsed"]
            ccolor = CALL_COLORS.get(ctype, "#6B7280")
            self.ax.text(panel_x, y, f"● Call {call['call_id']}: {ctype.upper()} @ N{call['location']}", ha="left", va="center", fontsize=8, color=ccolor, zorder=11)
            self.ax.text(panel_x + 8, y, f"{elapsed}min ({tone})", ha="left", va="center", fontsize=7, color="#94A3B8", zorder=11)

        # Unit status panel (bottom left)
        self.ax.text(1, -2.5, "UNITS", ha="left", va="center", fontsize=10, color="#06B6D4", fontweight="bold", zorder=11)
        for i, unit in enumerate(state["units"]):
            y = -3.5 - i * 0.9
            status = unit["status"]
            ucolor = UNIT_COLORS.get(status, "#FFFFFF")
            call_info = f" -> Call {unit['current_call']}" if unit.get("current_call") else ""
            self.ax.text(1, y, f"U{unit['unit_id']}: ", ha="left", va="center", fontsize=8, color="#94A3B8", zorder=11)
            self.ax.text(3, y, f"{status.upper()}{call_info}", ha="left", va="center", fontsize=8, color=ucolor, zorder=11)

        # Recent events banner (center, ephemeral)
        events = state.get("recent_events", [])
        if events:
            event_text = events[-1][:50]
            self.ax.text(12, -2, event_text, ha="center", va="center", fontsize=9, color="#F59E0B", alpha=0.8, zorder=11, style="italic")

        # Hospital status (bottom right)
        self.ax.text(20, -2.5, "HOSPITALS", ha="left", va="center", fontsize=10, color="#EC4899", fontweight="bold", zorder=11)
        for i, h in enumerate(state.get("hospital_statuses", [])):
            y = -3.5 - i * 0.9
            status = h["reported_status"]
            hcolor = "#22C55E" if status == "accepting" else "#EF4444"
            self.ax.text(20, y, f"H{h['hospital_id']}: ", ha="left", va="center", fontsize=8, color="#94A3B8", zorder=11)
            self.ax.text(23, y, status.upper(), ha="left", va="center", fontsize=8, color=hcolor, zorder=11)


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="episode.json", help="Episode JSON from export_episode.py")
    parser.add_argument("--output", default="dispatchr_demo.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--frames-per-step", type=int, default=5, help="Interpolation frames per env step (lower=faster render)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        episode_data = json.load(f)

    print(f"Loaded {len(episode_data)} steps from {args.input}")
    renderer = EpisodeRenderer(episode_data, args.output, args.fps, args.frames_per_step)
    renderer.render()


if __name__ == "__main__":
    main()
