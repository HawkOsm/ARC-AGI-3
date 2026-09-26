"""SHAPES — which objects belong together, judged by shape the way a person would at a glance.

For every pair of distinct object kinds on screen (a kind = colour + exact shape), ShapeRelations lists:
    identical      same shape, different colour                    (the grey L and the yellow block in ar25)
    turned         same shape after rotating and/or flipping
    scaled         one is the other drawn 2x, 3x or 4x bigger
    fills-gap      one object's cells fill a gap (notch, hole) of the other: key and lock, piece and slot
    fits-inside    one fits into an enclosed hole of the other
    same-colour    different shapes in the same colour: probably one group
Nothing here knows any game. The work grows with the number of distinct kinds, not with the number of pairs:
identical copies (tiles, dots) are one kind, compared once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .look import ScreenReader
from .scene import COLOUR_NAMES, LevelScene, SceneObject

RELATION_STRENGTH = {"identical": 1.0, "turned": 0.8, "fills-gap": 0.8, "scaled": 0.6, "fits-inside": 0.5,
                     "same-colour": 0.3}
MAX_KINDS = 30  # the largest this many kinds are compared (a screen of 200 tiles has few kinds)
MAX_PER_OBJECT = 3  # each object keeps its strongest few relations (a frame "holds" everything otherwise)
SNUG = 2.0  # fits-inside: the hole is at most this many times the piece's area


@dataclass(slots=True)
class Correspondence:
    relation: str  # a RELATION_STRENGTH key
    a: int  # object id (the representative of its kind)
    b: int
    strength: float

    def describe(self) -> str:
        words = {"identical": "same shape", "turned": "same shape, turned or flipped", "scaled": "same shape, scaled",
                 "fills-gap": "fills a gap in", "fits-inside": "fits inside", "same-colour": "same colour as"}
        return f"#{self.a} {words[self.relation]} #{self.b}"


def mask_of(obj: SceneObject) -> np.ndarray:
    h, w, raw = obj.shape
    return np.frombuffer(raw, dtype=bool).reshape(h, w)


def turns(mask: np.ndarray) -> list[np.ndarray]:
    """The 8 rotations/reflections of a mask."""
    out = []
    for m in (mask, np.fliplr(mask)):
        for k in range(4):
            out.append(np.rot90(m, k))
    return out


def same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


class ShapeRelations:
    """Pairwise shape correspondences between the kinds on screen (a utility class)."""

    @staticmethod
    def find(scene: LevelScene) -> list[Correspondence]:
        kinds: dict[tuple, SceneObject] = {}
        for o in scene.objects:
            if o.background or scene.root(o.id) != o.id:
                continue
            kinds.setdefault((o.colour, o.shape), o)
        reps = sorted(kinds.values(), key=lambda o: -o.size)[:MAX_KINDS]
        masks = {o.id: mask_of(o) for o in reps}
        found: list[Correspondence] = []
        for i, a in enumerate(reps):
            for b in reps[i + 1:]:
                relation = ShapeRelations._relate(a, b, masks[a.id], masks[b.id])
                if relation:
                    found.append(Correspondence(relation, a.id, b.id, RELATION_STRENGTH[relation]))
        found.sort(key=lambda c: -c.strength)
        per_object: dict[int, int] = {}
        kept = []
        for c in found:
            if per_object.get(c.a, 0) >= MAX_PER_OBJECT or per_object.get(c.b, 0) >= MAX_PER_OBJECT:
                continue
            per_object[c.a] = per_object.get(c.a, 0) + 1
            per_object[c.b] = per_object.get(c.b, 0) + 1
            kept.append(c)
        return kept

    @staticmethod
    def _relate(a: SceneObject, b: SceneObject, ma: np.ndarray, mb: np.ndarray) -> str | None:
        if a.size > 1 and b.size > 1:
            if same_mask(ma, mb):
                return "identical" if a.colour != b.colour else None  # same colour + shape = copies, one kind
            if any(same_mask(t, mb) for t in turns(ma)):
                return "turned"
            if ShapeRelations._scaled(ma, mb) or ShapeRelations._scaled(mb, ma):
                return "scaled"
        if ShapeRelations._fills_gap(ma, mb) or ShapeRelations._fills_gap(mb, ma):
            return "fills-gap"
        if ShapeRelations._fits_inside(ma, mb) or ShapeRelations._fits_inside(mb, ma):
            return "fits-inside"
        if a.colour == b.colour and a.size > 1 and b.size > 1:
            return "same-colour"
        return None

    @staticmethod
    def _scaled(small: np.ndarray, big: np.ndarray) -> bool:
        for k in (2, 3, 4):
            if big.shape == (small.shape[0] * k, small.shape[1] * k):
                down = big.reshape(small.shape[0], k, small.shape[1], k).mean(axis=(1, 3)) >= 0.5
                if any(same_mask(t, down) for t in turns(small)):
                    return True
        return False

    @staticmethod
    def _gaps(mask: np.ndarray) -> list[np.ndarray]:
        """Connected empty regions inside a mask's box (notches and holes), each cropped to its own box."""
        empty = ~mask
        gaps = []
        for blob in ScreenReader.changed_blobs(empty):
            cells = np.array(blob)
            if len(cells) == empty.size:
                continue
            y0, x0 = cells.min(axis=0)
            y1, x1 = cells.max(axis=0)
            crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
            crop[cells[:, 0] - y0, cells[:, 1] - x0] = True
            touches_edge = y0 == 0 or x0 == 0 or y1 == mask.shape[0] - 1 or x1 == mask.shape[1] - 1
            gaps.append((crop, touches_edge))
        return gaps  # type: ignore[return-value]

    @staticmethod
    def _fills_gap(piece: np.ndarray, holder: np.ndarray) -> bool:
        """An irregular piece (not a plain rectangle: those fill every rectangular gap) matching a gap exactly."""
        if piece.sum() < 3 or piece.all() or holder.size <= piece.size:
            return False
        return any(any(same_mask(t, gap) for t in turns(piece)) for gap, _edge in ShapeRelations._gaps(holder))

    @staticmethod
    def _fits_inside(piece: np.ndarray, holder: np.ndarray) -> bool:
        if piece.sum() < 2:
            return False
        for gap, touches_edge in ShapeRelations._gaps(holder):
            if not touches_edge and gap.shape[0] >= piece.shape[0] and gap.shape[1] >= piece.shape[1] \
                    and piece.sum() <= gap.sum() <= SNUG * piece.sum() and not same_mask(gap, piece):
                return True
        return False


def colour_name(obj: SceneObject) -> str:
    return COLOUR_NAMES[obj.colour]
