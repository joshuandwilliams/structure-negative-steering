#!/usr/bin/env python3
"""Draw the NegativeSteering hex logo and favicon.

Matches the sticker style used by CDALabeller and AutoCDAScorer: a black
hexagon, a green rim, the package name arched over the top in white, and one
motif below it. The palette is sampled from those two logos rather than chosen
again, so the three sit together.

The motif is what negative steering does. A receptor holds its shape while the
effector moves off the decoy surface it was predicted onto and onto the true
one. The original pose is drawn hollow and the steered pose solid, with an arc
between them, so the before and after read in one glance at 16 px.

    python3 docs/make_logo.py

Writes docs/logo.png (240 px, for the README) and docs/favicon.ico (a 16/32/
48/64/128/256 px stack, so a browser picks its own size rather than scaling
the 240 down).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent

# Sampled from CDALabeller/man/figures/logo.png and AutoCDAScorer's.
INK = "#000000"       # hexagon fill
RIM = "#4c6433"       # hexagon border
RECEPTOR = "#d8d677"  # the chain that stays put
EFFECTOR = "#e28060"  # the chain that moves
WHITE = "#fbfbfb"

R = 1.0               # hexagon circumradius, in axis units
RIM_W = 7.0


def hexagon(r: float) -> np.ndarray:
    """Flat-topped hexagon, vertices at the left and right as the others use."""
    a = np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def arched_text(ax, text: str, radius: float, y0: float, size: float,
                spread_deg: float = 104.0) -> None:
    """Set `text` along a circular arc, each glyph rotated to the tangent.

    Matplotlib has no text-on-a-path, so each glyph is placed and rotated by
    hand. The angular step is proportional to each glyph's own advance width,
    measured from its outline, rather than one step per character. Even steps
    leave visible gaps around narrow letters, which is what makes hand-arched
    text look wrong, and this name has four of them.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    fp = FontProperties(size=size, weight="bold")

    def advance(s: str) -> float:
        if s == " ":
            return TextPath((0, 0), "n", prop=fp).get_extents().width * 0.6
        return TextPath((0, 0), s, prop=fp).get_extents().width

    # Kerning is ignored, so pairs are measured together and the single-glyph
    # widths scaled to match the whole string. Without this the arc runs long.
    widths = [advance(c) for c in text]
    gap = advance("nn") - 2 * advance("n")
    total = sum(widths) + gap * (len(text) - 1)
    steps = [(w + gap) / total * spread_deg for w in widths]

    ang = 90.0 + spread_deg / 2.0
    for ch, step in zip(text, steps):
        centre = ang - step / 2.0
        rad = math.radians(centre)
        ax.text(radius * math.cos(rad), y0 + radius * math.sin(rad), ch,
                ha="center", va="center", color=WHITE, fontsize=size,
                fontweight="bold", rotation=centre - 90.0,
                rotation_mode="anchor", zorder=7)
        ang -= step


def blob(cx: float, cy: float, rx: float, ry: float, rot: float = 0.0,
         lobes: int = 5, wobble: float = 0.05, phase: float = 0.0) -> np.ndarray:
    """A closed rounded outline, as a stand-in for a folded domain.

    A plain ellipse reads as a ball, and a cartoon backbone trace turns to mush
    at 16 px. This is an ellipse with a few very shallow lobes, which is enough
    to look like a protein surface and still survives the favicon.
    """
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    r = 1.0 + wobble * np.sin(lobes * t + phase)
    x, y = rx * r * np.cos(t), ry * r * np.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return np.column_stack([cx + x * c - y * s, cy + x * s + y * c])


def draw(ax, *, with_text: bool, ribs: bool = True,
         lw_scale: float = 1.0) -> None:
    ax.add_patch(Polygon(hexagon(R), closed=True, facecolor=INK,
                         edgecolor=RIM, linewidth=RIM_W * lw_scale,
                         joinstyle="miter", zorder=1))

    # The receptor sits low and wide. Both effector poses dock onto its top
    # edge, so the move reads as sideways along one surface rather than as two
    # unrelated shapes.
    ax.add_patch(Polygon(blob(0.0, -0.46, 0.68, 0.25, phase=1.1),
                         closed=True, facecolor=RECEPTOR, edgecolor="none",
                         zorder=3))
    # Ribs, so the receptor reads as a folded chain rather than a puddle. They
    # are the first thing to close up when the canvas shrinks, so the small
    # favicon frames leave them out rather than carry a smear.
    if ribs:
        for dx in (-0.33, -0.11, 0.11, 0.33):
            ax.plot([dx, dx], [-0.59, -0.35], color=INK, solid_capstyle="round",
                    linewidth=2.1 * lw_scale, zorder=4)

    # Original pose: hollow, sitting off to one side of the true surface.
    ax.add_patch(Polygon(blob(-0.36, -0.04, 0.26, 0.22, phase=0.4),
                         closed=True, facecolor="none", edgecolor=EFFECTOR,
                         linewidth=3.6 * lw_scale, zorder=5))
    # Steered pose: solid, seated on the true surface.
    ax.add_patch(Polygon(blob(0.33, -0.10, 0.26, 0.22, phase=0.4),
                         closed=True, facecolor=EFFECTOR, edgecolor="none",
                         zorder=5))

    # The arrow carries the whole idea, so it is drawn long enough to read as
    # a path from one pose to the other rather than as a mark between them.
    ax.add_patch(FancyArrowPatch(
        (-0.34, 0.24), (0.31, 0.20),
        connectionstyle="arc3,rad=-0.34", color=WHITE,
        linewidth=3.2 * lw_scale, zorder=6,
        arrowstyle="-|>,head_length=0.62,head_width=0.34",
        mutation_scale=18 * lw_scale, shrinkA=0, shrinkB=0,
        capstyle="round"))

    if with_text:
        # Font size is in points too, so it scales with the canvas like the
        # stroke widths do, or the name shrinks when the drawing is resized.
        arched_text(ax, "NegativeSteering", radius=0.72, y0=0.0,
                    size=13.2 * lw_scale, spread_deg=106.0)

    ax.set_xlim(-R * 1.03, R * 1.03)
    ax.set_ylim(-R * 1.03, R * 1.03)
    ax.set_aspect("equal")
    ax.axis("off")


# Everything is drawn on one canvas this size and then resized down. Line
# widths and arrow heads are in points, which are absolute, so drawing a 16 px
# figure directly gives a logo whose strokes are twenty times too heavy for it.
# Draw once, big, and let the resampler do the shrinking.
BASE_PX = 512


def render(px: int, *, with_text: bool, ribs: bool = True):
    from PIL import Image
    dpi = 100
    fig = plt.figure(figsize=(BASE_PX / dpi, BASE_PX / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_alpha(0.0)
    draw(ax, with_text=with_text, ribs=ribs, lw_scale=BASE_PX / 240.0)
    tmp = HERE / ".logo_tmp.png"
    fig.savefig(tmp, dpi=dpi, transparent=True)
    plt.close(fig)
    im = Image.open(tmp).convert("RGBA").resize((px, px), Image.LANCZOS)
    tmp.unlink()
    return im


def main() -> None:
    render(240, with_text=True).save(HERE / "logo.png")


    # The name is unreadable below 256 px and the ribs close up below 64, so
    # the small frames drop each in turn. A browser picks the frame nearest the
    # size it wants, so each one is drawn for that size rather than scaled.
    frames = []
    for px in (16, 32, 48, 64, 128, 256):
        frames.append(render(px, with_text=px >= 256, ribs=px >= 64))
    frames[-1].save(HERE / "favicon.ico", format="ICO",
                    sizes=[(f.width, f.height) for f in frames],
                    append_images=frames[:-1])
    print(f"wrote {HERE / 'logo.png'} and {HERE / 'favicon.ico'}")


if __name__ == "__main__":
    main()
