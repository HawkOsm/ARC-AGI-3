"""The model (agent/brain/world.py): records, enums, tallies, the per-level map."""
import numpy as np

from agent.brain.world import (
    LEVEL_UP,
    RESET_ACTION,
    Action,
    ClickInfo,
    CounterMemory,
    GameMode,
    Goal,
    LevelMemory,
    Screen,
    Tally,
)


def test_action_is_a_tuple():
    assert Action(1) == (1, -1, -1)
    assert hash(Action(6, 3, 4)) == hash((6, 3, 4))
    assert {Action(6, 3, 4): "x"}[(6, 3, 4)] == "x"
    assert Action(6, 3, 4).is_click and not Action(2).is_click
    assert RESET_ACTION == (0, -1, -1)


def test_click_info_unpacks():
    prior, colour = ClickInfo(0.3, 7)
    assert (prior, colour) == (0.3, 7)


def test_game_mode_from_actions():
    assert GameMode.from_actions([1, 2, 3, 4]) == GameMode.MOVER
    assert GameMode.from_actions([6]) == GameMode.CLICKER
    assert GameMode.from_actions([6, 7]) == GameMode.CLICKER
    assert GameMode.from_actions([1, 2, 3, 4, 5, 6]) == GameMode.MIXED
    assert GameMode.from_actions([3, 4, 6, 7]) == GameMode.MIXED
    assert GameMode.from_actions([5, 7]) == GameMode.OTHER


def test_tally_chance_moves_from_prior_to_evidence():
    t = Tally()
    assert t.chance(0.5) == 0.5
    for _ in range(20):
        t.add(False)
    assert t.chance(0.5) < 0.1


def test_goal_confidence_uses_prior_only_without_evidence():
    g = Goal(feature=("count", 3), direction=1, support=0, against=0, source="cue", prior=0.45)
    assert g.confidence() == 0.45
    g.support = 3
    assert g.confidence() == (3 + 1) / (3 + 2)


def test_counter_memory_hides_ignored_cells_and_restarts_on_new_shape():
    c = CounterMemory((4, 4))
    c.ignore[0, 0] = True
    grid = np.ones((4, 4), dtype=np.uint8)
    assert c.hide(grid)[0, 0] == 255 and c.hide(grid)[1, 1] == 1
    c.hide(np.ones((5, 5), dtype=np.uint8))
    assert c.shape == (5, 5) and c.version == 1 and not c.ignore.any()


def _map_with_chain() -> LevelMemory:
    """screens 1 -(A1)-> 2 -(A2)-> 3 -(A3)-> LEVEL_UP, and 1 -(A4)-> 3 as a shortcut"""
    m = LevelMemory()
    for sid in (1, 2, 3):
        m.screens[sid] = Screen(sid, {}, None, b"")
    m.record_result(1, Action(1), 2)
    m.record_result(2, Action(2), 3)
    m.record_result(1, Action(4), 3)
    m.record_result(3, Action(3), LEVEL_UP)
    return m


def test_level_map_finds_shortest_known_route():
    m = _map_with_chain()
    assert m.has_solution
    distance, _ = m.walking_distances(1)
    assert distance == {1: 0, 2: 1, 3: 1}
    assert m.route_to_level_up(1) == [(Action(4), 3), (Action(3), LEVEL_UP)]


def test_level_map_counts_contradictions():
    m = _map_with_chain()
    assert m.record_result(1, Action(1), 3) is False  # was 2
    assert m.contradictions == 1


def test_rename_screens_merges_screens_that_now_look_the_same():
    m = _map_with_chain()
    for sid, s in m.screens.items():
        s.snapshot = b"same" if sid in (2, 3) else b"other"
    m.rename_screens(lambda snap: 20 if snap == b"same" else 10)
    assert set(m.screens) == {10, 20}
    assert m.screens[10].results[Action(1)] == 20
