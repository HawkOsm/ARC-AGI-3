"""Hypothesis mode: scene tracking (agent/brain/scene.py) and checking the LLM's predictions (llm.Lab)."""
import numpy as np

from agent.brain.llm import Hypothesis, Lab, parse_hypothesis
from agent.brain.scene import Change, Observation, RelationFinder, SceneTracker
from agent.brain.world import Action, World


def _screen(l_at: int) -> np.ndarray:
    """A 20x20 screen: blue background, a light-blue mirror line at x=9-10, a grey 3x3 block at x=l_at and its
    black mirror image, plus two white dots on the black block (they move with it)."""
    g = np.full((20, 20), 9, dtype=np.uint8)
    g[:, 9:11] = 10
    g[5:8, l_at:l_at + 3] = 4
    mx = 19 - (l_at + 2)
    g[5:8, mx:mx + 3] = 5
    g[5, mx] = g[5, mx + 2] = 0
    return g


def _tracker() -> SceneTracker:
    return SceneTracker(World(shape=(20, 20)))


def test_mirror_and_movement_are_seen():
    t = _tracker()
    t.update(0, _screen(13), None)
    objects = t.scene(0).objects
    facts = RelationFinder.find(objects, 20, 20)
    assert any("mirrored across" in f for f in facts)
    obs = t.update(0, _screen(14), Action(4))
    moves = {(c.dx, c.dy) for c in obs.changes if c.kind == "moved"}
    assert (1, 0) in moves and (-1, 0) in moves  # grey went right, its mirror image went left


def test_twins_keep_their_ids_when_they_move_together():
    t = _tracker()
    t.update(0, _screen(13), None)
    dots_before = sorted(o.id for o in t.scene(0).objects if o.colour == 0)
    t.update(0, _screen(14), Action(4))
    dots_after = sorted(o.id for o in t.scene(0).objects if o.colour == 0)
    assert dots_before == dots_after


def test_parts_that_always_move_together_become_one_object():
    t = _tracker()
    t.update(0, _screen(12), None)
    for x in (13, 14, 15):
        obs = t.update(0, _screen(x), Action(4))
    scene = t.scene(0)
    black = next(o.id for o in scene.objects if o.colour == 5)
    assert len(scene.members(black)) == 2  # the two dots
    assert len([c for c in obs.changes if c.kind == "moved"]) == 2  # grey + the black composite, not 4


def test_prediction_check():
    h = Hypothesis("rule", [1], "goal", {}, Action(4), 3, "moved", "left")
    left = Observation(Action(4), [Change(3, "moved", -3, 0)])
    right = Observation(Action(4), [Change(3, "moved", 3, 0)])
    assert Lab.check(h, left).startswith("CORRECT")
    assert Lab.check(h, right).startswith("WRONG")
    nothing = Hypothesis("rule", [], "goal", {}, Action(5), 0, "nothing", "none")
    assert Lab.check(nothing, Observation(Action(5), [])).startswith("CORRECT")


def test_parse_hypothesis():
    text = ('{"rule": "grey mirrors black", "controlled": [4], "goal": "#4 covers #5", '
            '"roles": [{"id": 4, "role": "player"}], '
            '"test": {"action": "ACTION3", "object": 4, "change": "moved", "direction": "left"}}')
    h = parse_hypothesis(text, [1, 2, 3, 4])
    assert h is not None and h.controlled == [4] and h.roles == {4: "player"} and h.test_action == Action(3)
    assert parse_hypothesis("not json", [1]) is None
    illegal = parse_hypothesis(text.replace("ACTION3", "ACTION6 1 1"), [1, 2, 3, 4])
    assert illegal is not None and illegal.test_action is None
