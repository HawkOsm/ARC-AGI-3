"""BRAIN — the orchestrator. One Brain plays one game.

Every step (Brain.next_action):
    1. LOOK     the frame arrives -> settled picture                    (look.ScreenReader)
    2. EVENTS   died? level up? new play?  (the level change is the only real reward)
    3. LEARN    first screen of a level? take a first look                (look.FirstLook)
                What did my last action do? update the beliefs          (learn.*Learner, goals.Goals)
    4. DECIDE   follow the LLM advisor's plan if one is on               (llm.Advisor)
                otherwise pick the next action from the beliefs           (decide.Explorer)

Long-term beliefs live in `self.world` (world.World). Short-term memory - what just happened
and the route being followed - lives in `self.memory` (world.WorkingMemory).
"""
from __future__ import annotations

import os
import random
from collections import defaultdict, deque

import numpy as np

from .decide import Explorer
from .goals import GoalBoard, Goals, ScreenFeatures
from .learn import ActionLearner, AvatarLearner, CounterLearner, DeathJudge, TouchLearner
from .llm import Lab, PromptWriter, advisor
from .look import FirstLook, ScreenReader
from .scene import RelationFinder, SceneTracker
from .settings import LLM_ASK_EVERY, LLM_MODE, LLM_RECENT_STEPS
from .trace import Tracer
from .world import (
    DIED,
    LEVEL_UP,
    NO_INFO,
    RESET_ACTION,
    Action,
    Death,
    LevelMemory,
    Screen,
    WorkingMemory,
    World,
    compress_grid,
    decompress_grid,
)


class Brain:
    def __init__(self, seed: int = 0, name: str = "") -> None:
        self.name = name  # the game id, for logs only
        self.rng = random.Random(seed)
        self.world = World()
        self.memory = WorkingMemory(recent=deque(maxlen=LLM_RECENT_STEPS))
        self.stats: defaultdict[str, int] = defaultdict(int)  # diagnostics only
        # the learners (the only writers of beliefs)
        self.first_look = FirstLook(self.world)
        self.counter_learner = CounterLearner(self.world)
        self.avatar_learner = AvatarLearner(self.world)
        self.touch_learner = TouchLearner(self.world)
        self.action_learner = ActionLearner(self.world)
        self.death_judge = DeathJudge(self.world)
        self.goals = Goals(self.world)
        # the deciders
        self.explorer = Explorer(self.world, self.memory, self.goals, self.rng, self.stats, self.restart_level)
        self.prompt = PromptWriter(self.world, self.memory, self.stats)
        self.tracer = Tracer.for_game(name)  # None unless ARC3_TRACE_DIR is set
        # hypothesis mode (ARC3_LLM_MODE=hypothesis): objects on screen + experiments the LLM proposes
        self.mode = os.environ.get("ARC3_LLM_MODE", LLM_MODE)
        self.scene = SceneTracker(self.world) if self.mode == "hypothesis" else None
        self.lab = Lab()
        self.goal_board = GoalBoard()
        self.goal_list: list = []  # the last list of possible goals shown to the LLM (for debug output)
        self._traced_verdicts = 0

    # ── public ──────────────────────────────────────────────────────────────

    def next_action(self, state: str, frame_layers, levels_completed: int, available: list[int],
                    full_reset: bool = False) -> Action:
        """The whole agent: given the latest frame, return (action_id, x, y)."""
        action = self._think(state, frame_layers, levels_completed, available, full_reset)
        if self.tracer is not None:
            verdicts = self.lab.verdicts
            check = self.lab.last_verdict if verdicts != self._traced_verdicts else ""
            self._traced_verdicts = verdicts
            self.tracer.record(state, ScreenReader.settled_grid(frame_layers), action, self.world, self.memory,
                               self.stats, check)
        self.stats["actions"] += 1
        self.memory.life_length = 0 if action == RESET_ACTION else self.memory.life_length + 1
        return action

    def forget_last_action(self) -> None:
        """The last action never reached the game (network error): learn nothing from it."""
        self.memory.last_action = None
        self.memory.route, self.memory.expected_screen = [], None

    def restart_level(self) -> Action:
        """RESET. Only ever sent after GAME_OVER or after steps were taken in this life:
        a RESET with 0 steps taken in the level restarts the WHOLE GAME."""
        m = self.memory
        self.world.avatar.under = None
        m.route, m.expected_screen, m.plan = [], None, []
        self._remember("RESET -> the level restarts")
        m.last_action, m.last_grid, m.came_by = RESET_ACTION, None, None
        m.at_level_start = True
        return RESET_ACTION

    # ── the loop ────────────────────────────────────────────────────────────

    def _think(self, state, frame_layers, level, available, full_reset) -> Action:
        m = self.memory
        m.why = ""
        # 1. LOOK
        grid = ScreenReader.settled_grid(frame_layers)
        # 2. EVENTS
        if state == "NOT_PLAYED" or (grid is None and state != "GAME_OVER"):
            m.why = "start the game"
            return self.restart_level()
        if state == "GAME_OVER":
            self._on_death()
            m.why = "game over: RESET the level"
            return self.restart_level()
        assert grid is not None
        if full_reset:
            self._on_new_play()
        if m.level is None:
            m.level = level
            m.at_level_start = True
        elif level > m.level or state == "WIN":
            self._on_level_up(level if level > m.level else m.level + 1)
            if state == "WIN":
                m.why = "the whole game is won"
                return self.restart_level()
        elif level < m.level:
            self._on_new_play()
            m.level = level
            m.at_level_start = True
        assert m.level is not None

        # 3. LEARN
        if self.action_learner.note_available(available):
            self.stats["mode_changes"] += 1
        level_map = self.world.levels.setdefault(m.level, LevelMemory())
        if not level_map.items:
            self.first_look.look(m.level, grid)
            self.goals.cues_from_first_look(m.level)
        self._learn_from_last_move(grid)
        screen_id = self._screen_id(grid)
        screen = self._remember_screen(level_map, screen_id, grid, available)
        if m.at_level_start:
            if level_map.start_screen is None:
                level_map.start_screen = screen_id
            m.at_level_start = False
        if m.last_action is not None and m.last_action != RESET_ACTION:
            self._record_move(level_map, screen_id)
        if m.route and m.expected_screen is not None and m.expected_screen != screen_id:
            self.stats["route_mismatches"] += 1  # never trust a plan the world disagrees with
            m.route = []
        self._note_last_step(grid, screen_id)
        self._watch_scene(grid)

        # 4. DECIDE
        action = self._experiment(level_map, grid, available) if self.scene is not None \
            else self._advice(level_map, grid, available)
        m.last_from_advisor = action is not None
        if action is None:
            action = self.explorer.choose(level_map, screen, available)
        m.screen_before_last_id = m.last_screen_id if m.last_action != RESET_ACTION else None
        m.last_grid, m.last_screen_id = grid, screen_id
        m.last_action = action
        return action

    def _remember(self, line: str) -> None:
        """One line of history ("ACTIONn -> what happened"), for the advisor and traces."""
        self.memory.recent.append(line)
        self.memory.last_outcome = line

    def _note_last_step(self, grid: np.ndarray, screen_id: int) -> None:
        """History line for the advisor; an advised action that changed nothing ends its plan."""
        m = self.memory
        if m.last_action is None or m.last_action == RESET_ACTION or m.last_grid is None:
            return
        hide = self.world.counters.hide
        self._remember(PromptWriter.describe_step(m.last_action, hide(m.last_grid), hide(grid), None))
        if m.last_from_advisor and screen_id == m.last_screen_id and m.plan:
            self.stats["llm_plans_dropped"] += 1  # the plan expected something; nothing happened
            m.plan = []

    def _watch_scene(self, grid: np.ndarray) -> None:
        """Hypothesis mode: track the objects; if the last action was the LLM's test, check its prediction."""
        m = self.memory
        if self.scene is None or m.level is None:
            return
        last = m.last_action if m.last_action != RESET_ACTION else None
        observation = self.scene.update(m.level, grid, last)
        pending = self.lab.pending
        if pending is not None and observation is not None and observation.action == pending.test_action:
            verdict = Lab.check(pending, observation)
            self.stats["llm_tests_correct" if verdict.startswith("CORRECT") else "llm_tests_wrong"] += 1
            self.lab.record(verdict)

    def _experiment(self, level_map: LevelMemory, grid: np.ndarray, available: list[int]) -> Action | None:
        """Hypothesis mode: every LLM_ASK_EVERY brain actions, ask the object question and run its test action."""
        m = self.memory
        adviser = advisor()
        if adviser is None or self.scene is None or m.level is None or level_map.has_solution \
                or self.lab.pending is not None or self.stats["actions"] < self.lab.next_ask_at:
            return None
        scene = self.scene.scene(m.level)
        wholes = [o for o in scene.objects if scene.root(o.id) == o.id]
        relations = scene.remember_relations(RelationFinder.find(wholes, *grid.shape))
        self.goal_list = self.goal_board.possible_goals(scene)
        question = self.prompt.object_question(scene, relations, list(scene.observations)[-12:], self.lab.history,
                                               available, self.goal_list, Goals.purposes(scene))
        self.stats["llm_questions"] += 1
        hypothesis = adviser.hypothesise(question, available, tag=self.name)
        self.lab.next_ask_at = self.stats["actions"] + LLM_ASK_EVERY
        if hypothesis is None or hypothesis.test_action is None:
            return None
        self.lab.latest = self.lab.pending = hypothesis
        self.goal_board.vote(self.goal_list, hypothesis.goal_choice)
        chosen = self.goal_board.chosen.describe() if self.goal_board.chosen else hypothesis.goal
        m.theory = f"{hypothesis.rule} | goal: {chosen} | controlled: {hypothesis.controlled}"
        m.why = f"LLM test: expects {hypothesis.prediction()}"
        m.route, m.expected_screen = [], None
        self.stats["llm_actions"] += 1
        return hypothesis.test_action

    def _advice(self, level_map: LevelMemory, grid: np.ndarray, available: list[int]) -> Action | None:
        """The advisor's next action, or None (advisor off, level already solved, no usable answer)."""
        m = self.memory
        adviser = advisor()
        if adviser is None or level_map.has_solution:
            m.plan = []
            return None
        if not m.plan and adviser.available() and self.stats["actions"] >= m.ask_again_at:
            self.stats["llm_questions"] += 1
            advice = adviser.propose(self.prompt.describe(grid, available), available, tag=self.name)
            m.plan = list(advice.plan)
            m.theory = advice.theory or m.theory
            if m.plan:
                self.stats["llm_plans"] += 1
                m.plan_size = len(m.plan)
            else:
                m.ask_again_at = self.stats["actions"] + 4
        while m.plan:
            action = m.plan.pop(0)
            if action[0] in available:
                m.route, m.expected_screen = [], None
                self.stats["llm_actions"] += 1
                m.why = f"LLM plan, step {m.plan_size - len(m.plan)} of {m.plan_size}"
                return action
        return None

    # ── learning from the last move ─────────────────────────────────────────

    def _learn_from_last_move(self, grid: np.ndarray) -> None:
        """Update counters (and re-identify all screens if they grew) and the avatar."""
        m, world = self.memory, self.world
        if m.last_grid is None or m.last_action in (None, RESET_ACTION):
            return
        assert m.last_action is not None
        before = world.counters.hide(m.last_grid)
        if self.counter_learner.learn(m.last_grid, grid):
            self.stats["counter_mask_updates"] += 1
            for level_map in world.levels.values():
                level_map.rename_screens(self._screen_id_from_snapshot)
            m.last_screen_id = self._screen_id(m.last_grid)
            m.screen_before_last_id = None
            m.route, m.expected_screen = [], None
            before = world.counters.hide(m.last_grid)
        after = world.counters.hide(grid)
        self.avatar_learner.learn(before, after, m.last_action[0])
        self.touch_learner.learn(before, after, m.last_action[0])

    def _record_move(self, level_map: LevelMemory, screen_id: int) -> None:
        """Last screen + last action -> this screen. Also teaches world.actions."""
        m = self.memory
        assert m.last_action is not None and m.last_screen_id is not None
        previous = level_map.screens.get(m.last_screen_id)
        if previous is None:
            return
        if not level_map.record_result(m.last_screen_id, m.last_action, screen_id):
            self.stats["contradictions"] += 1
            m.route = []
        changed = screen_id != m.last_screen_id
        undid = None
        if m.screen_before_last_id is not None and m.screen_before_last_id != m.last_screen_id:
            undid = screen_id == m.screen_before_last_id
        self.action_learner.record(m.last_action, previous.candidates.get(m.last_action, NO_INFO),
                                   changed, False, previous.avatar_at, undid, m.came_by)
        m.came_by = m.last_action if changed else m.came_by

    # ── screen identity ─────────────────────────────────────────────────────

    def _screen_id(self, grid: np.ndarray) -> int:
        """What makes a screen THIS screen: the picture minus counters."""
        return hash(self.world.counters.hide(grid).tobytes())

    def _screen_id_from_snapshot(self, snapshot: bytes) -> int:
        return self._screen_id(decompress_grid(snapshot, self.world.counters.shape))

    def _remember_screen(self, level_map: LevelMemory, screen_id: int, grid: np.ndarray,
                         available: list[int]) -> Screen:
        screen = level_map.screens.get(screen_id)
        if screen is not None:
            return screen
        world = self.world
        candidates = self.explorer.candidate_actions(grid, available)
        screen = Screen(screen_id, candidates, world.avatar.position(world.counters.hide(grid)), compress_grid(grid))
        screen.features = ScreenFeatures.of(world, grid)
        level_map.screens[screen_id] = screen
        self.stats["screens"] += 1
        return screen

    # ── events ──────────────────────────────────────────────────────────────

    def _on_death(self) -> None:
        m, world = self.memory, self.world
        self.stats["deaths"] += 1
        if self.lab.pending is not None:
            self.lab.record("the test action LOST A LIFE")
        m.plan = []
        if m.last_action not in (None, RESET_ACTION):
            assert m.last_action is not None
            self._remember(PromptWriter.describe_step(m.last_action, None, None, "you lost a life; the level restarts"))
        level_map = world.levels.get(m.level) if m.level is not None else None
        if level_map is None or m.last_screen_id is None or m.last_action in (None, RESET_ACTION):
            return
        assert m.last_action is not None and m.level is not None
        previous = level_map.screens.get(m.last_screen_id)
        deaths_before = len(world.deaths)
        if self.death_judge.is_timeout(m.level, m.life_length):
            self.stats["timeouts"] += 1 + deaths_before - len(world.deaths)
            return
        level_map.record_result(m.last_screen_id, m.last_action, DIED)
        if m.last_grid is not None:
            self.touch_learner.learn(world.counters.hide(m.last_grid), None, m.last_action[0], outcome=DIED)
        if previous is not None:
            self.action_learner.record(m.last_action, previous.candidates.get(m.last_action, NO_INFO),
                                       True, True, previous.avatar_at, None, None)
            world.deaths.append(Death(m.level, m.life_length, previous, m.last_action, previous.avatar_at))

    def _on_level_up(self, new_level: int) -> None:
        m, world = self.memory, self.world
        self.stats["level_ups"] += 1
        if self.lab.pending is not None:
            self.lab.record("the test action FINISHED THE LEVEL")
        if self.scene is not None and m.level is not None:
            self.goal_board.learn_from_win(self.scene.scene(m.level))
            self.goal_board.chosen = None
        m.plan = []
        if m.last_action not in (None, RESET_ACTION):
            assert m.last_action is not None
            self._remember(PromptWriter.describe_step(m.last_action, None, None, "LEVEL COMPLETE; a new level starts"))
        m.life_length = 0
        level_map = world.levels.get(m.level) if m.level is not None else None
        if level_map is not None and m.last_screen_id is not None \
                and m.last_action not in (None, RESET_ACTION):
            assert m.last_action is not None and m.level is not None
            level_map.record_result(m.last_screen_id, m.last_action, LEVEL_UP)
            if m.last_grid is not None:
                self.touch_learner.learn(world.counters.hide(m.last_grid), None, m.last_action[0],
                                         outcome=LEVEL_UP)
            winning_screen = level_map.screens.get(m.last_screen_id)
            if winning_screen is not None:
                self.goals.learn_from_win(m.level, level_map, winning_screen)
                self.stats["goals_known"] = len(world.goals)
        m.level = new_level
        world.avatar.under = None
        m.last_action, m.last_grid, m.came_by = None, None, None
        m.route, m.expected_screen = [], None
        m.at_level_start = True

    def _on_new_play(self) -> None:
        """A full reset happened (back to level 1)."""
        self.memory.plan = []
        self.memory.level = 0
        self.memory.at_level_start = True
