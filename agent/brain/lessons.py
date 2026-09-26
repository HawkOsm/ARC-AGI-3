"""LESSONS — what the game taught so far, carried from one level to the next (hypothesis mode).

    ColourStory     what objects of one colour did on earlier levels: moved by keys, selected by a click, ...
    WinLesson       how a finished level was finished: which object ended on which, what the keys controlled
    GameLessons     the game's lessons: news for the new level ("NEW: never seen before", "CHANGED: never moved
                    before, now the keys move it"), win lessons for the LLM, and won colour pairs for the goal list
Colour is the identity across levels: shapes change from level to level, colours usually keep their meaning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .control import ControlMap
from .scene import COLOUR_NAMES, LevelScene

MIN_SIZE = 3  # smaller objects (dots, marks) are not reported as news


@dataclass(slots=True)
class ColourStory:
    levels: set[int] = field(default_factory=set)
    moved_by_keys: bool = False
    selected_by_click: bool = False
    roles: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WinLesson:
    level: int
    actions: int
    mover_colour: int | None  # the object brought onto the target when the level ended, if the agent was steering
    target_colour: int | None
    relation: str
    controls: list[str]

    def describe(self) -> str:
        text = f"Level {self.level + 1} was finished after {self.actions} actions"
        if self.mover_colour is not None and self.target_colour is not None:
            text += (f": the level ended when a {COLOUR_NAMES[self.mover_colour]} object was brought onto a "
                     f"{COLOUR_NAMES[self.target_colour]} object ({self.relation})")
        if self.controls:
            text += ". Controls then: " + " | ".join(self.controls)
        return text + "."


class GameLessons:
    """Everything learned on finished levels (one per game; written at each level's end)."""

    def __init__(self) -> None:
        self.colours: dict[int, ColourStory] = {}
        self.wins: list[WinLesson] = []
        self.won_pairs: dict[tuple[int, int], int] = {}  # (mover colour, target colour) -> levels won this way

    def end_level(self, level: int, scene: LevelScene, controls: ControlMap, actions: int,
                  mover: int | None, target: int | None, relation: str, roles: dict[int, str]) -> WinLesson:
        colour_of = {o.id: o.colour for o in scene.objects}
        self._remember_colours(level, scene, controls, roles)
        mc, tc = colour_of.get(mover) if mover is not None else None, \
            colour_of.get(target) if target is not None else None
        lesson = WinLesson(level, actions, mc, tc, relation, controls.describe(scene)[:4])
        self.wins.append(lesson)
        if mc is not None and tc is not None:
            self.won_pairs[(mc, tc)] = self.won_pairs.get((mc, tc), 0) + 1
        return lesson

    def won_targets(self) -> set[int]:
        return {t for _m, t in self.won_pairs}

    def _remember_colours(self, level: int, scene: LevelScene, controls: ControlMap, roles: dict[int, str]) -> None:
        now = self.behaviour(scene, controls)
        for o in scene.objects:
            if o.background or o.size < MIN_SIZE:
                continue
            story = self.colours.setdefault(o.colour, ColourStory())
            story.levels.add(level)
            moved, selected = now.get(o.colour, (False, False))
            story.moved_by_keys |= moved
            story.selected_by_click |= selected
            if o.id in roles:
                story.roles.add(roles[o.id])

    @staticmethod
    def behaviour(scene: LevelScene, controls: ControlMap) -> dict[int, tuple[bool, bool]]:
        """colour -> (moved by keys in some mode, selected by clicking it) on this level so far."""
        colour_of = {o.id: o.colour for o in scene.objects}
        out: dict[int, tuple[bool, bool]] = {}
        for mode in controls.modes:
            for obj in mode.steps:
                if obj in colour_of:
                    c = colour_of[obj]
                    out[c] = (True, out.get(c, (False, False))[1])
            for obj in mode.clicked:
                if obj in colour_of:
                    c = colour_of[obj]
                    out[c] = (out.get(c, (False, False))[0], True)
        return out

    def news(self, scene: LevelScene, controls: ControlMap) -> list[str]:
        """What is new or behaves differently on this level, compared with the finished ones."""
        if not self.colours:
            return []
        now = self.behaviour(scene, controls)
        by_colour: dict[int, list[int]] = {}
        for o in scene.objects:
            if not o.background and o.size >= MIN_SIZE and scene.root(o.id) == o.id:
                by_colour.setdefault(o.colour, []).append(o.id)
        lines = []
        for colour, ids in sorted(by_colour.items()):
            name, which = COLOUR_NAMES[colour], " ".join(f"#{i}" for i in ids[:6])
            story = self.colours.get(colour)
            if story is None:
                lines.append(f"NEW: {which} ({name}) - this colour never appeared on earlier levels; find out "
                             f"what it is for")
                continue
            moved, selected = now.get(colour, (False, False))
            if moved and not story.moved_by_keys:
                lines.append(f"CHANGED: {which} ({name}) never moved on earlier levels; now the keys move it")
            if selected and not story.selected_by_click:
                lines.append(f"CHANGED: {which} ({name}) can now be selected by clicking it")
            count_before = len(story.levels)
            if len(ids) > 1 and count_before:
                lines.append(f"{which} ({name}): {len(ids)} of them now" +
                             (f"; earlier: {', '.join(sorted(story.roles))}" if story.roles else ""))
        return lines[:10]

    def describe_wins(self) -> list[str]:
        return [w.describe() for w in self.wins[-4:]]
