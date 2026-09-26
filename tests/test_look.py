"""The reader (agent/brain/look.py): objects, changed blobs, shapes, the first look."""
import numpy as np

from agent.brain.look import FirstLook, ScreenReader
from agent.brain.world import LevelMemory, World


def _grid() -> np.ndarray:
    g = np.zeros((8, 8), dtype=np.uint8)
    g[1:3, 1:3] = 4  # a 2x2 block of colour 4
    g[5:7, 5:7] = 4  # its twin
    g[1:3, 5:7] = 9  # the same shape in colour 9
    return g


def test_settled_grid_takes_the_last_layer():
    layers = [np.zeros((2, 2)), np.ones((2, 2))]
    assert ScreenReader.settled_grid(layers).tolist() == [[1, 1], [1, 1]]
    assert ScreenReader.settled_grid([]) is None


def test_find_objects_returns_same_colour_blobs():
    objects = ScreenReader.find_objects(_grid())
    sizes = sorted((o.colour, o.size) for o in objects)
    assert sizes == [(0, 52), (4, 4), (4, 4), (9, 4)]
    block = next(o for o in objects if o.colour == 9)
    assert block.bbox == (1, 5, 2, 6)
    y, x = block.centre_cell
    assert _grid()[y, x] == 9  # the centre cell is always on the object


def test_find_objects_skips_invisible_cells():
    visible = np.ones((8, 8), dtype=bool)
    visible[1:3, 1:3] = False
    colours = [o.colour for o in ScreenReader.find_objects(_grid(), visible=visible)]
    assert colours.count(4) == 1


def test_changed_blobs_are_8_connected():
    changed = np.zeros((5, 5), dtype=bool)
    changed[0, 0] = changed[1, 1] = True  # diagonal neighbours: one blob
    changed[4, 4] = True
    assert sorted(len(b) for b in ScreenReader.changed_blobs(changed)) == [1, 2]


def test_shape_is_position_independent():
    a, b = [o for o in ScreenReader.find_objects(_grid()) if o.colour == 4]
    assert ScreenReader.shape_of(a) == ScreenReader.shape_of(b)


def test_first_look_catalogues_kinds_and_patterns():
    world = World(shape=(8, 8))
    world.levels[0] = LevelMemory()
    FirstLook(world).look(0, _grid())
    assert len(world.levels[0].items) == 4
    kinds = {(k.colour, k.size) for k in world.kinds.values()}
    assert (4, 4) in kinds and (9, 4) in kinds
    types = {p.type for p in world.patterns}
    assert types == {"repeated", "same_shape"}
