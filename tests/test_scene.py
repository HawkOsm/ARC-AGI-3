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


def test_relations_seen_earlier_are_remembered():
    from agent.brain.scene import LevelScene
    scene = LevelScene()
    assert scene.remember_relations(["#5 and #6: same shape"]) == ["#5 and #6: same shape"]
    assert scene.remember_relations([]) == ["#5 and #6: same shape (seen earlier in this level)"]


def test_goal_board_ranks_same_shape_pair_first_and_follows_llm_votes():
    from agent.brain.goals import GoalBoard
    t = _tracker()
    g = _screen(13)
    g[14:17, 2:5] = 11  # a yellow block with the grey block's exact shape
    t.update(0, g, None)
    board = GoalBoard()
    goals = board.possible_goals(t.scene(0))
    grey = next(o.id for o in t.scene(0).objects if o.colour == 4)
    yellow = next(o.id for o in t.scene(0).objects if o.colour == 11)
    assert {goals[0].a, goals[0].b} == {grey, yellow} and goals[0].relation == "identical"
    board.vote(goals, 1)
    assert board.chosen is goals[0]
    assert board.possible_goals(t.scene(0))[0].score > goals[0].score  # the pick moved it up


def test_navigator_plans_the_shortest_key_sequence_and_avoids_blocked_moves():
    from agent.brain.steer import Navigator
    steps = {1: (-3, 0), 2: (3, 0), 3: (0, 3), 4: (0, -3)}
    plan = Navigator.plan((0, 0), (6, 9), steps, (3, 3), (20, 20), set())
    assert sorted(plan) == [2, 2, 3, 3, 3]
    blocked = {((0, 0), 2)}  # can't go down from the start: go right first
    plan = Navigator.plan((0, 0), (6, 9), steps, (3, 3), (20, 20), blocked)
    assert plan[0] == 3 and sorted(plan) == [2, 2, 3, 3, 3]
    unreachable = Navigator.plan((0, 0), (7, 9), steps, (3, 3), (20, 20), set())  # 7 is not a multiple of 3
    assert sorted(unreachable) == [2, 2, 3, 3, 3]  # gets as near as it can


def _scene_with(*objects):
    from agent.brain.scene import LevelScene, SceneObject
    scene = LevelScene()
    for oid, colour, size in objects:
        scene.objects.append(SceneObject(oid, colour, size, (0, 0, 2, 2), (3, 3, b"")))
    return scene


def test_control_map_learns_selection_modes_and_switches():
    from agent.brain.control import ControlMap
    scene = _scene_with((1, 10, 30), (2, 4, 9), (3, 0, 1))  # a line, a block, a one-cell marker
    controls = ControlMap()
    assert controls.observe(scene, Observation(Action(3), [Change(1, "moved", -3, 0)])) == "key"
    assert controls.current == 0
    # a switch: only the marker jumped (and a composite's parts jumped different ways)
    assert controls.observe(scene, Observation(Action(5), [Change(3, "moved", 15, 0)])) == "switch"
    assert controls.current is None
    controls.observe(scene, Observation(Action(1), [Change(2, "moved", 0, -3)]))
    assert controls.current == 1 and controls.modes[1].entered_by == [Action(5)]
    controls.observe(scene, Observation(Action(4), [Change(2, "moved", 3, 0)]))  # same mode, another key
    assert controls.current == 1 and controls.steps(1, [2]) == {1: (-3, 0), 4: (0, 3)}
    # clicking the line selects it again: the known mode comes back (same key, same object, same way)
    controls.observe(scene, Observation(Action(6, 1, 1), [Change(3, "moved", -15, 0)], clicked=1))
    controls.observe(scene, Observation(Action(3), [Change(1, "moved", -3, 0)]))
    assert controls.current == 0 and controls.modes[0].clicked == [1] and controls.clicks_select()
    assert controls.best_mode([2]) == 1


def test_a_game_without_selection_has_one_mode():
    from agent.brain.control import ControlMap
    scene = _scene_with((1, 4, 9))
    controls = ControlMap()
    for key, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (1, (0, -1))):
        controls.observe(scene, Observation(Action(key), [Change(1, "moved", dx, dy)]))
    controls.observe(scene, Observation(Action(4), []))  # blocked: nothing moved, nothing learned
    assert len(controls.modes) == 1 and set(controls.steps(0, [1])) == {1, 2, 3}


def test_lessons_report_new_and_changed_objects_and_won_colour_pairs():
    from agent.brain.control import ControlMap
    from agent.brain.lessons import GameLessons
    lessons = GameLessons()
    level1 = _scene_with((1, 10, 30), (2, 4, 9), (5, 11, 9))
    c1 = ControlMap()
    c1.observe(level1, Observation(Action(3), [Change(2, "moved", -3, 0)]))
    lessons.end_level(0, level1, c1, 20, 2, 5, "identical", {})
    assert lessons.won_pairs == {(4, 11): 1}
    assert "brought onto" in lessons.describe_wins()[0]
    level2 = _scene_with((1, 10, 30), (2, 4, 9), (5, 11, 9), (6, 11, 9), (7, 8, 12))
    c2 = ControlMap()
    c2.observe(level2, Observation(Action(3), [Change(1, "moved", -3, 0)]))  # the line moves now
    news = "\n".join(lessons.news(level2, c2))
    assert "NEW: #7" in news and "CHANGED: #1" in news and "2 of them" in news


def test_goal_board_keeps_reached_goals_and_prefers_won_colours():
    from agent.brain.goals import GoalBoard
    t = _tracker()
    g = _screen(13)
    g[14:17, 2:5] = 11
    t.update(0, g, None)
    scene = t.scene(0)
    grey = next(o.id for o in scene.objects if o.colour == 4)
    yellow = next(o.id for o in scene.objects if o.colour == 11)
    board = GoalBoard()
    before = board.possible_goals(scene)[0].score
    board.won_pairs[(4, 11)] = 1
    assert board.possible_goals(scene)[0].score == 6 * before  # (1 + 2 x won pair) x (1 + won target colour)
    board.done.append((grey, yellow))
    assert all(grey not in (c.a, c.b) for c in board.possible_goals(scene))
