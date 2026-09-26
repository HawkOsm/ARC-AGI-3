"""Brain hints for the Duck harness (agent/duck/hints.py): our findings, in the Duck's words and coordinates."""
import time

import numpy as np

from agent.duck.hints import BrainHints, _names, hints_for, to_action
from agent.brain.world import Action


def test_duck_action_labels_become_our_actions():
    assert to_action("UP") == Action(1) and to_action("space") == Action(5)
    assert to_action("MOUSE(row=4, col=7)") == Action(6, 7, 4)  # row = y, col = x
    assert to_action("RESET") is None


def test_colour_names_are_renamed_once_and_keys_use_the_ducks_labels():
    assert _names(["#3 (light blue): ACTION1 up 3"]) == ["#3 (sky blue (S)): UP up 3"]
    assert _names(["#2 (blue)"]) == ["#2 (blue (b))"]


def _play_ar25(labels: list[str]) -> list[tuple[str, np.ndarray, int]]:
    import arc_agi
    from arcengine import GameAction
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("ar25", scorecard_id=arc.open_scorecard(tags=["test"]))
    frame = env.reset()
    history = [("", np.asarray(frame.frame[-1], dtype=np.uint8), frame.levels_completed + 1)]
    for label in labels:
        action = to_action(label)
        assert action is not None
        data = {"x": action[1], "y": action[2]} if action[0] == 6 else {}
        frame = env.step(GameAction.from_id(action[0]), data=data)
        history.append((label, np.asarray(frame.frame[-1], dtype=np.uint8), frame.levels_completed + 1))
    return history


def test_hints_from_real_ar25_moves_name_the_controls_and_the_shape_goal():
    history = _play_ar25(["UP", "DOWN", "LEFT", "RIGHT", "RIGHT"])
    hints = BrainHints()
    started = time.time()
    hints.feed(history)
    report = hints.report()
    assert time.time() - started < 2.0
    assert any("UP" in line and "LEFT" in line for line in report["control_modes"])
    assert report["possible_goals"] and "#" in report["possible_goals"][0]
    assert all(" rows " in where and " cols " in where for where in report["objects"].values())
    # incremental: feeding the same history again adds nothing and gives the same report
    hints.feed(history)
    assert hints.report() == report


def test_hints_never_raise():
    assert "error" in hints_for("broken", [("UP", "not a grid", 1)])  # type: ignore[list-item]
