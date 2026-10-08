"""Rig horn geometry the Skyblivion mesh weights rigidly to one torso bone.

A horn is the source bone's vertices on one side past a width and height;
its base is the centroid of its lowest vertices and its tip that of the
vertices farthest from the base.  A vertex's position along the horn is its
distance from the base over the horn's length (0..1); joints sit at fixed
fractions (knots) and the vertex weight moves down the chain those knots name.

See: skyb_retarget/README.md#the-horns
"""
from dataclasses import dataclass

import numpy as np

#: Half-width (in horn fraction) of the blend between two neighboring chain bones.
BLEND = 0.05


@dataclass
class Horn:
    """One horn: where it is, what it is cut from, and its (bone, fraction) chain."""
    side: float
    source: str
    select: tuple
    knots: list
    base: np.ndarray = None
    length: float = 1.0

    def mask(self, mesh) -> np.ndarray:
        """Which of `mesh`'s vertices belong to this horn."""
        min_x, min_z = self.select
        v = mesh.verts
        return ((mesh.column(self.source) > 0.5) & (v[:, 0] * self.side > min_x)
                & (v[:, 2] > min_z))

    def fraction(self, verts) -> np.ndarray:
        """Each vertex's position along the horn, 0 at the base, 1 at the tip."""
        d = np.linalg.norm(verts - self.base, axis=1) / self.length
        return np.clip(d, 0.0, 1.0)


def fit_horn(mesh, side: float, source: str, select, knots) -> Horn:
    """Measure one horn on the whole mesh."""
    horn = Horn(side, source, tuple(select), list(knots))
    sel = mesh.verts[horn.mask(mesh)]
    horn.base = sel[sel[:, 2] <= sel[:, 2].min() + 4.0].mean(0)
    dist = np.linalg.norm(sel - horn.base, axis=1)
    tip = sel[dist >= np.percentile(dist, 95)].mean(0)
    horn.length = float(np.linalg.norm(tip - horn.base))
    return horn


def joints(horn: Horn, mesh) -> dict:
    """{chain bone: centroid of the horn's cross-section at its knot}."""
    sel = mesh.verts[horn.mask(mesh)]
    t = horn.fraction(sel)
    return {bone: sel[np.abs(t - f) < 0.04].mean(0)
            for bone, f in horn.knots[1:]}


def chain_weights(t: np.ndarray, knots) -> np.ndarray:
    """(V, len(knots)) weights: the bone whose knot precedes t, blended near the next knot."""
    at = np.array([k for _b, k in knots])
    w = np.zeros((len(t), len(knots)))
    seg = np.clip(np.searchsorted(at, t, side='right') - 1, 0, len(knots) - 1)
    w[np.arange(len(t)), seg] = 1.0
    for i in range(1, len(knots)):
        near = np.abs(t - at[i]) < BLEND
        mix = (t[near] - at[i] + BLEND) / (2 * BLEND)
        w[near] = 0.0
        w[near, i - 1] = 1.0 - mix
        w[near, i] = mix
    return w


def reweight(names: list, w: np.ndarray, mesh, horns) -> tuple:
    """Spread horn vertices' source-bone weight down each horn's chain; returns (names, w).

    The chain's first bone is the one the source bone already maps to, so the
    weight leaves it and lands on the chain by position along the horn.
    """
    names, w = list(names), w.copy()
    for horn in horns:
        bones = [b for b, _k in horn.knots]
        for b in bones:
            if b not in names:
                names.append(b)
                w = np.c_[w, np.zeros(len(w))]
        mask = horn.mask(mesh)
        cols = [names.index(b) for b in bones]
        moved = mesh.column(horn.source)[mask]
        w[mask, cols[0]] -= moved
        w[np.ix_(mask, cols)] += chain_weights(horn.fraction(mesh.verts[mask]),
                                               horn.knots) * moved[:, None]
    return names, w
