"""Put a retargeted limb's tip where the source animation puts its own.

Per frame and chain: the source tip's offset from the source chain root,
taken in a body bone's frame and scaled by the chains' rest length ratio, is
added to the target chain root; a FABRIK solve from the current (retargeted)
pose moves the target joints onto it and each bone turns to its solved
segment, keeping its twist.  Arms and horns of a different build keep the
source's spread: tips the source never crosses do not cross.

See: skyb_retarget/README.md#arm-reference-pose
"""
import numpy as np

from asset_convert.havok.clip_retarget import Skeleton, mat_to_quat_wxyz, world_positions
from skyb_retarget.leg_ik import aim, fabrik, locals_at


def _offset(world, root: int, tip: int, ref: int) -> np.ndarray:
    """Tip minus root, in bone `ref`'s frame."""
    return (world[tip][3, :3] - world[root][3, :3]) @ np.linalg.inv(world[ref][:3, :3])


class ReachRig:
    """Index tables, length ratios and stance offsets for `chains` (root, ..., tip names)."""

    def __init__(self, src: Skeleton, dst: Skeleton, chains, frame_bone: str,
                 stance=None):
        """`frame_bone` carries the offsets; `stance` is (source idle worlds, blend).

        With a stance, a tip's offset is the source's scaled movement away
        from its idle, added to a blend of the target's rest offset (0) and
        the source idle's scaled offset (1).
        """
        self.src, self.dst = src, dst
        self.ref = src.index[frame_bone]
        self.chains = []
        for names in chains:
            idx = [dst.index[n] for n in names]
            rest = _offset(dst.world, idx[0], idx[-1], self.ref)
            k = float(np.linalg.norm(rest)
                      / np.linalg.norm(_offset(src.world, idx[0], idx[-1], self.ref)))
            bias = np.zeros(3)
            if stance is not None:
                idle = _offset(stance[0], idx[0], idx[-1], self.ref) * k
                bias = (1.0 - stance[1]) * (rest - idle)
            self.chains.append({'idx': idx, 'k': k, 'bias': bias})

    def target(self, chain: dict, src_world, dst_world) -> np.ndarray:
        """World spot this frame's chain tip should reach."""
        root, tip = chain['idx'][0], chain['idx'][-1]
        off = _offset(src_world, root, tip, self.ref) * chain['k'] + chain['bias']
        return dst_world[root][3, :3] + off @ dst_world[self.ref][:3, :3]


def _reach(rig: ReachRig, chain: dict, local, world, src_world) -> np.ndarray:
    """Solve one chain onto its target and turn its bones; returns new worlds."""
    idx = chain['idx']
    solved = fabrik(np.array([world[i][3, :3] for i in idx]),
                    rig.target(chain, src_world, world))
    for n, (bone, child) in enumerate(zip(idx, idx[1:])):
        world = aim(local, rig.dst, world, bone, child, solved[n + 1])
    return world


def reach_tips(tracks: list, clip, rig: ReachRig) -> None:
    """Rewrite the chain bones' rotations in `tracks` (in place) frame by frame."""
    by_name = {tr.bone: tr for tr in tracks}
    for f in range(len(clip.times)):
        src_world = world_positions(clip, rig.src, f)
        local = locals_at(by_name, rig.dst, f)
        world = rig.dst.fk(local)
        for chain in rig.chains:
            world = _reach(rig, chain, local, world, src_world)
        for chain in rig.chains:
            for i in chain['idx'][:-1]:
                by_name[rig.dst.names[i]].rotations[f] = mat_to_quat_wxyz(
                    local[i][:3, :3])
