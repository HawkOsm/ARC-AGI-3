# LLM vs Algorithm Decision Split in Game-Playing Agents

> Research report — evidence only, no recommendations or rankings.
> Compiled 2026-09-26. Extends the existing survey at `docs/research/COMMUNITY.md`.

---

## 1) Sources Checked

| # | URL | Status | Notes |
|---|-----|--------|-------|
| 1 | https://arcprize.org/blog/arc-prize-2026-milestone-1 | read | Milestone #1 blog post (winners) |
| 2 | https://github.com/DriesSmit/ARC3-solution | read | Tufa Labs / Duck repo |
| 3 | https://arxiv.org/abs/2608.14490 | read | Twin paper (HTML version) |
| 4 | https://arxiv.org/abs/2607.28287 | read (partial) | Tycho paper (HTML truncated before full results tables) |
| 5 | https://arxiv.org/abs/2607.01531 | read | OPINE-World paper (HTML version) |
| 6 | https://arxiv.org/abs/2605.25931 | read | Explore Before You Solve (HTML version) |
| 7 | https://arxiv.org/abs/2607.15439 | read | Do Coding Agents Need Executable World Models? (HTML) |
| 8 | https://arxiv.org/abs/2605.05138 | read | Executable World Models for ARC-AGI-3 (HTML) |
| 9 | https://arxiv.org/abs/2609.21032 | read | Scaling Discovery through Test-Time Communication |
| 10 | https://arxiv.org/abs/2606.18542 | snippet | CodeARC benchmark |
| 11 | https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore | read | Just Explore repo |
| 12 | https://github.com/wd13ca/ARC-AGI-3-Agents | read | Blind Squirrel repo |
| 13 | https://github.com/FluxonApps/arc-prize | read | Fluxonian repo |
| 14 | https://github.com/TesslaRay/tomas-engine-arc-agi-3 | read | TOMAS Engine repo |
| 15 | https://github.com/GameDevGitHub/GuidedRandomAgent | read | GuidedRandom repo |
| 16 | https://github.com/dhanaabhirajk/ARC-AGI-3-Agents | read | Play Zero repo |
| 17 | https://arxiv.org/abs/2402.01817 | read | LLM-Modulo / Kambhampati (HTML) |
| 18 | https://arxiv.org/abs/2304.11477 | snippet | LLM+P |
| 19 | https://arxiv.org/abs/2204.01691 | read | SayCan (HTML) |
| 20 | https://arxiv.org/abs/2305.16291 | read | Voyager (HTML) |
| 21 | https://arxiv.org/abs/2210.03629 | read | ReAct (HTML) |
| 22 | https://arxiv.org/abs/2303.11366 | read | Reflexion (HTML) |
| 23 | https://arxiv.org/abs/2301.12050 | snippet | DECKARD |
| 24 | https://arxiv.org/abs/2302.06692 | snippet | ELLM |
| 25 | https://arxiv.org/abs/2305.16653 | snippet | AdaPlanner |
| 26 | https://arxiv.org/abs/2209.07753 | read | Code as Policies (HTML) |
| 27 | https://arxiv.org/abs/2402.12275 | read | WorldCoder (HTML) |
| 28 | https://arxiv.org/abs/2405.15383 | snippet | GIF-MCTS |
| 29 | https://arxiv.org/abs/2603.03305 | snippet | Constrained decoding effects |
| 30 | https://arxiv.org/abs/2505.20139 | read | StructEval (structured output consistency) |
| 31 | Qwen3/3.5 blog and benchmarks | snippet | Via search |
| 32 | Gemma-4 technical report | snippet | Via search |
| 33 | Phi-4-mini benchmarks | snippet | Via search |
| 34 | Reki team write-ups | failed | No repos or detailed write-ups found beyond milestone blog |
| 35 | forge team write-ups | failed | No repos or detailed write-ups found beyond milestone blog |

---

## 2) Part A — ARC-AGI-3 Agents

### A1. Tufa Labs "The Duck" (Milestone #1 Winner, 1st Place)

**Source**: https://arcprize.org/blog/arc-prize-2026-milestone-1, https://github.com/DriesSmit/ARC3-solution
**Model**: Qwen 3.6 27B FP8
**LLM calls per action**: ~1 (LLM drives every action via REPL)
**Score**: Won 1st place Milestone #1 ($37.5K)
**Status**: read

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Both | LLM perceives via modalities (images, ASCII, segmentation tool) provided by code | "perceive the board (through multiple modalities including rendered images, raw ASCII grids, and a segmentation tool for region-specific zooming)" |
| Object segmentation | Both | Code provides segmentation tool; LLM uses it | "a segmentation tool for region-specific zooming" |
| Tracking what changed | LLM | LLM observes changes through rendered frames in context | not stated explicitly |
| Rule/world-model hypothesis | LLM | LLM writes and tests Python code to model the game | "treats ARC-AGI-3 games as interactive coding problems" |
| Goal hypothesis | LLM | LLM infers the goal through interaction | not stated explicitly |
| Choosing next action | LLM | LLM writes Python code that calls action functions | "uses a Python REPL to interact with the environment" |
| Multi-step planning | LLM | LLM writes code for multi-step plans | not stated explicitly |
| Exploration strategy | LLM | LLM decides what to explore via code | not stated explicitly |
| Verifying hypothesis | LLM | LLM can test hypotheses by writing and running code | implied by REPL approach |
| Memory across levels | Code | Oldest messages evicted when context fills | "the system continuously pops the oldest messages to keep the agent under its context limit, so it can keep playing indefinitely" |
| When to reset | not stated | | |
| When to call the LLM | Code | LLM is called for every action step | implied by architecture |

**Ablation**: The Milestone #1 blog states: "In all three cases, teams found that hand-crafted tools actually hurt the model; letting it improvise worked better." No exact numerical ablation reported for Duck specifically.

---

### A2. Reki (Milestone #1, 2nd Place)

**Source**: https://arcprize.org/blog/arc-prize-2026-milestone-1
**Model**: Gemma 4 31B
**LLM calls per action**: 1 (one JSON action per call)
**Score**: 2nd place Milestone #1
**Status**: read (blog only; no repo or detailed write-up found)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Both | Code renders frames as labeled images; VLM interprets | "each turn it renders the recent frames as labeled images, feeds them to Gemma-4-31B locally" |
| Object segmentation | Code | Identifies small, rare-colored, button-like shapes | "Hardcoded rules make fallback and exploratory clicks prefer small, rare-colored, button-like shapes instead of random pixels." |
| Tracking what changed | LLM | VLM outputs JSON describing changes | "asks for a single JSON object describing what changed, a short plan, and the next 1-4 actions." |
| Rule/world-model hypothesis | not stated | | |
| Goal hypothesis | not stated | | |
| Choosing next action | Both | VLM returns JSON plan; code has fallback heuristic clicks | "returns one JSON action per step... there is a numpy click heuristic mechanism (no GPU needed)." |
| Multi-step planning | LLM | VLM outputs short plan and 1-4 actions per call | "asks for a single JSON object describing what changed, a short plan, and the next 1-4 actions." |
| Exploration strategy | Code | Dead-signature stops wasting clicks on inert objects | "A 'Dead-signature' notices when clicking a type of object never changes anything and stops wasting clicks on it for the rest of the level." |
| Verifying hypothesis | not stated | | |
| Memory across levels | Code | Reflection memory refreshed periodically | "It also keeps a running reflection memory (refreshed every ~10 steps)." |
| When to reset | not stated | | |
| When to call the LLM | not stated | | |

**Ablation**: None reported for Reki specifically.

---

### A3. forge / Md Boktiar Mahbub Murad (Milestone #1, 3rd Place)

**Source**: https://arcprize.org/blog/arc-prize-2026-milestone-1
**Model**: Gemma 4 31B
**LLM calls per action**: 1
**Score**: 3rd place Milestone #1
**Status**: read (blog only; no repo found)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Both | Code renders frames to images for VLM | "render frames to images, show them to a locally-served Gemma-4-31B" |
| Object segmentation | not stated | | |
| Tracking what changed | not stated | | |
| Rule/world-model hypothesis | not stated | | |
| Goal hypothesis | not stated | | |
| Choosing next action | LLM | Gemma returns structured JSON actions with guards | "return structured JSON actions with repair and legal-action guards." |
| Multi-step planning | Both | Optional generator/arbiter mechanism (turned off in best run) | "The submission has a generator for multiple candidate actions and an arbiter to score and pick between them" |
| Exploration strategy | Both | Optional confidence prompt for safe moves (turned off in best run) | "an optional confidence prompt that forces safe/reversible moves when the model is unsure." |
| Verifying hypothesis | not stated | | |
| Memory across levels | Both | Maintains reflection memory (turned off in best run) | "keep a reflection memory, and return structured JSON actions" |
| When to reset | not stated | | |
| When to call the LLM | not stated | | |

**Ablation**: The best run turned off ALL extra machinery: memory bank, structured reasoning prompt, multi-step lookahead, generator/arbiter, confidence prompt. "the best run had all of the extra machinery turned off—no memory bank, no structured reasoning prompt, no multi-step lookahead—just the model looking at frames and picking actions" (read, verbatim). Exact scores not provided.

**Cross-winner ablation** (all three Milestone #1 winners): "In all three cases, teams found that hand-crafted tools actually hurt the model; letting it improvise worked better." (read, verbatim from blog)

---

### A4. Twin (arXiv 2608.14490)

**Source**: https://arxiv.org/abs/2608.14490
**Model**: Codex GPT-5.6 Sol (extended-high reasoning effort)
**LLM calls per action**: Not stated explicitly; 224k tokens per action on average
**Status**: read (HTML version)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | LLM | Reads integer grid (no rendered pixels) | "Figure 15 shows what the agent actually perceives each step: no rendered pixels, a page of integers." |
| Object segmentation | not stated | | |
| Tracking what changed | Code | Harness detects five progress signals | "It ranks reachable states by five signs of progress relative to the current frame: a color appears or disappears, a compact region changes..." |
| Rule/world-model hypothesis | LLM | Writes Python world model (digital twin) | "Twin uses a coding agent to write the world model in code at test time." |
| Goal hypothesis | Both | Harness generates candidates; LLM writes predicate | "The harness, not the coding agent, generates one candidate per progress signal... The coding agent then edits goal_reached to describe the top candidate's salient change." |
| Choosing next action | Code | Harness enforces plan via executor after replay-validation | "Every scored action starts from a replay-validated twin." |
| Multi-step planning | Code | BFS search within twin | "bounded breadth-first search (BFS), once inside the final twin and once inside the real engine." |
| Exploration strategy | Code | Harness generates candidates, prefers shorter paths | "The harness, not the coding agent, generates one candidate per progress signal and prefers shorter paths." |
| Verifying hypothesis | Code | Harness blocks action until replay passes | "The harness enforces that an action is not made until the program reproduces every previous observed game transition." |
| Memory across levels | not stated | | |
| When to reset | not stated | | |
| When to call the LLM | Code | Harness control loop triggers LLM for model writing/repair | "Twin repeatedly fits an executable world model to a growing interaction log, plans inside it, and executes while checking the prediction at each step." |

**Ablation** (RHAE on 183 public levels):

| Configuration | RHAE | Games fully cleared | Tokens per scored action |
|---|---|---|---|
| Twin (full: validate–explore–plan + Codex GPT-5.6 Sol xhigh) | 93.3 | 23 | 224k |
| Off-the-shelf Codex without validate–explore–plan loop | 61.1 | 13 | 48k |
| Direct play (base model, no coding) | 7.8 | not stated | not stated |
| EWM (Rodionov, arXiv 2605.05138) baseline | 63.8 | not stated | 715k |

"Twin clears 179 out of 183 levels (97.8%), and does so more efficiently than humans in 158 out of 179 levels (88.3%)." (read, verbatim)

**Model comparisons** (from Twin paper): "With GPT-5.6 Sol, Prime Agent scores 78.3... With Claude Opus 5, it scores 95.5... Twin reaches 93.3 with Sol." (read, verbatim)

The key code-side machinery (the harness) accounts for the difference from 61.1 to 93.3 RHAE — a +32.2 point gain from having code control exploration, verification, and planning. Twin uses 4.7× more tokens per action than the no-harness baseline (224k vs 48k) but is 3.1× less expensive than EWM (224k vs 715k).

---

### A5. Tycho (arXiv 2607.28287)

**Source**: https://arxiv.org/abs/2607.28287
**Model**: GPT-5.6 Sol, Claude Opus 4.8 / Opus 5
**LLM calls per action**: Varies; actor may delegate to model-builder agent
**Status**: read (HTML, partially truncated before full results tables)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Code | Environments formalized as deterministic Moore machines; code separates actionable from intermediate frames | "Separates actionable observations from intermediate frames" |
| Object segmentation | not stated | | |
| Tracking what changed | not stated | | |
| Rule/world-model hypothesis | LLM | Coding agent constructs game-specific programmatic world models | "a coding-agent that constructs game-specific models during interaction" |
| Goal hypothesis | LLM | Inferred during model construction | not stated explicitly |
| Choosing next action | Both | Actor decides when to delegate to model builder vs act directly | "Actor-requested delegation to a model builder obtains the highest observed mean RHAE" |
| Multi-step planning | Code | Planning against validated world model | not stated explicitly |
| Exploration strategy | LLM | "Active abstraction" — LLM decides when to invest in model building | "We call this joint problem active abstraction: generating a testable model from costly interaction and deciding when acquiring or using it is worth its cost." |
| Verifying hypothesis | Code | Automated replay verification | implied by architecture |
| Memory across levels | LLM | Meta-reflection for cross-level memory | "Meta-reflection turns trajectories into candidate system changes... World models make limited experience reusable." |
| When to reset | not stated | | |
| When to call the LLM | Both | Actor LLM decides when to delegate to model-builder LLM; code orchestrates | "Actor-requested delegation" |

**Ablation** (four policies tested):
1. No model (act directly)
2. Single-actor modeling
3. Actor-requested delegation
4. Automatically triggered repair (Orchestrator)

The Orchestrator policy was selected as best. Actor-requested delegation: **88.49 mean RHAE** (highest among reported).
"With this selected policy, GPT-5.6 Sol and Opus 5 both reach **100.00 RHAE** and complete all 183 levels." (read, verbatim)
Claude Opus 5 uses **14.5% fewer actions** than GPT-5.6 Sol at the same RHAE. (read)

Token telemetry: "the released bundle strips the per-call token telemetry... so the total cannot yet be independently reconstructed." (read)

The key insight: the LLM deciding WHEN to build a model (vs always or never) is itself the critical ablation. Exact RHAE for always-build and never-build policies not visible in truncated HTML.

---

### A6. OPINE-World (arXiv 2607.01531)

**Source**: https://arxiv.org/abs/2607.01531
**Model**: Claude Opus 4.8
**LLM calls per action**: Multiple (two cooperating agents: actor + synthesizer)
**Status**: read (HTML version)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Both | Object-centric programmatic world models learned online | "learns object-centric programmatic world models online from interaction" |
| Object segmentation | LLM | LLM learns object ontology; "ontology error" quantifies uncertainty | "Steers exploration via a Bayesian measure called 'ontology error' that quantifies uncertainty about the object model" |
| Tracking what changed | Code | Replay verification tracks all transitions | implied by CEGIS approach |
| Rule/world-model hypothesis | LLM | Synthesizer agent writes code with CEGIS | "Control and model construction are separated across two LLM-driven roles that share one symbolic world model." |
| Goal hypothesis | LLM | Fitted after first level cleared | "OPINE-World fits its predicate only after a first level is cleared." |
| Choosing next action | LLM | Actor agent uses model to plan | "The actor plans with it, the synthesizer repairs it, and the critic audits it." |
| Multi-step planning | LLM | Actor plans with world model | "replay verification and model-based planning" |
| Exploration strategy | Code | Bayesian ontology error measure guides where to explore | "The Bayesian diagnostic tracks row-level abstraction adequacy... ontology error... is the compact signal that directs where each looks next." |
| Verifying hypothesis | Both | Critic (LLM) audits replay-consistency of symbolic world model | "The critic instead scores the replay-consistency of a symbolic world model." |
| Memory across levels | not stated | | |
| When to reset | not stated | | |
| When to call the LLM | Code | Called when ontology error is high or counterexample found | implied by CEGIS loop |

**Results**: "OPINE-World solves 20 of 25 games without per-game training and reaches an action-efficiency score of 78.4 against the human baseline." (read, verbatim)

**Ablation**: Specific ablation table comparing with/without ontology error or with/without CEGIS not found in the available text.

---

### A7. Explore Before You Solve / AERA (arXiv 2605.25931)

**Source**: https://arxiv.org/abs/2605.25931
**Model**: Qwen2.5-0.5B / Qwen2.5-1.5B (notably tiny)
**LLM calls per action**: not stated
**Status**: read (HTML version)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | not stated | | |
| Object segmentation | not stated | | |
| Tracking what changed | not stated | | |
| Rule/world-model hypothesis | LLM | Free-form LLM hypothesis (noted as weakness) | "Stronger world model (future work): replace free-form LLM hypothesis with a structured DSL verified against observations." |
| Goal hypothesis | not stated | | |
| Choosing next action | Both | Constrained by architectural phases | "A three-phase architecture (EXPLORE, VERIFY, PLAN)... structured action constraints prevent invalid moves." |
| Multi-step planning | not stated | | |
| Exploration strategy | Code | Empirical budget threshold (~40% of human baseline) | "The adaptive exploration budget — approximately 40% of the human baseline — is an empirical heuristic fit on a small ablation." |
| Verifying hypothesis | Both | VERIFY phase refines hypothesis | "TRM's recursive refinement is structurally analogous to AERA's VERIFY phase." |
| Memory across levels | not stated | | |
| When to reset | not stated | | |
| When to call the LLM | Code | Architecture gates phase transitions via entropy | "AERA's entropy-gated mode switch never crosses the commitment threshold, the PLAN phase is never entered." |

**Ablation**:
- AERA (Qwen2.5-0.5B): **RHAE = 0.2116**, 4/25 games solved
- Random baseline: **RHAE = 0.0000**, 0/25 games solved
- No-explore baseline: **RHAE = 0.0000**, 0/25 games solved
- With exploration (0.5B model): **RHAE = 0.5290**

(read, verbatim: "AERA... achieving RHAE=0.2116 (4/25 solved) on these 25 games with Qwen2.5-0.5B, while random and no-explore baselines score 0.0000.")

This demonstrates even a 0.5B model contributes meaningfully when code structures the exploration.

---

### A8. Do Coding Agents Need Executable World Models? (arXiv 2607.15439)

**Source**: https://arxiv.org/abs/2607.15439
**Model**: GPT-5.4, GPT-5.5, GPT-5.6-sol
**LLM calls per action**: Varies by variant
**Status**: read (HTML version)

This paper IS an ablation study comparing four variants of world-model usage:

| Variant | Description |
|---------|-------------|
| Verification | Executable model + automated replay verification |
| Executable | Executable model, no verification |
| Textual | Text-only reasoning, no code model |
| Flexible-interface | Agent chooses its approach |

| Decision | Verification variant | Textual variant | Quote (verbatim) |
|----------|---------------------|-----------------|-------------------|
| Rule/world-model | LLM writes code | LLM reasons in text only | "An executable world model is a persistent, agent-authored hypothesis about the current task environment embodied in runnable code." |
| Choosing next action | LLM | LLM | "All substantive game-solving work is performed by Codex... and chooses all ordinary game actions." |
| Multi-step planning | LLM (uses model to anticipate) | LLM (text reasoning) | "A model-based agent can use an environment hypothesis to anticipate the consequences of candidate actions before taking them." |
| Verifying hypothesis | Code (automated replay) | LLM (text) | "The verification variant's workspace still contains plan_executor.py... executes one action in the real environment, compares the prediction with the actual result, and aborts the remaining sequence on any mismatch." |
| When to reset | Code | Code | "The controller initializes the game, sends RESET through the client when the current level must be restarted, and selects the next prompt sequence." |
| When to call LLM | Code | Code | "A thin external controller sends prompts to Codex and manages the game lifecycle." |

**Key ablation findings**:
1. "Requiring an executable deliverable is not universally beneficial: textual outperforms flexible-interface executable in both gpt-5.5 conditions." (read, verbatim)
2. "With gpt-5.6-sol, the verification variant completes every public level at xhigh and max with about 99% human-relative action efficiency while using fewer than half the human baseline's total actions." (read, verbatim)
3. The textual variant at max effort: "41% fewer actions than the human baseline" — completing every level.

**Detailed results** (average RHAE by model/effort across all four variants):
- GPT-5.4 (high): 34.36
- GPT-5.4 (xhigh): 48.05
- GPT-5.5 (high): 58.50
- GPT-5.5 (xhigh): 72.52

**Textual vs Executable vs Verification** (per-variant RHAE):

| Variant | GPT-5.5 high | GPT-5.5 xhigh | GPT-5.6-sol xhigh | GPT-5.6-sol max |
|---------|-------------|---------------|-------------------|-----------------|
| Verification | not stated | 74.78 | 98.97 | 98.77 |
| Flexible-interface Executable | 51.16 | 69.70 | not stated | not stated |
| Textual | 58.85 | 72.51 | 92.34 | 95.97 |

**Token costs** (GPT-5.6-sol):
- Verification (xhigh): 90.75M cost tokens
- Textual (xhigh): 32.32M cost tokens
- Verification (max): 103.49M cost tokens
- Textual (max): 30.60M cost tokens

For GPT-5.5 (the weaker model), textual reasoning WITHOUT code world models OUTPERFORMED executable models.

---

### A9. Executable World Models for ARC-AGI-3 (arXiv 2605.05138)

**Source**: https://arxiv.org/abs/2605.05138
**Model**: GPT-5.5 (high reasoning effort)
**LLM calls per action**: Multiple (iterative model refactoring)
**Status**: read (HTML version)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| Perception | Code | Predefined interfaces capture game state | "uses predefined interfaces" |
| Object segmentation | not stated | | |
| Tracking what changed | Code | Observation history maintained by harness | implied |
| Rule/world-model hypothesis | LLM | Coding agent writes and refactors Python world model | "maintains and refactors an executable Python world model based on observations" |
| Goal hypothesis | not stated | | |
| Choosing next action | Code | Planning against validated model | "verifies the model against all past observations before planning" |
| Multi-step planning | Code | Planning using world model | implied |
| Exploration strategy | not stated | | |
| Verifying hypothesis | Code | Automated verification against all past observations | "verifies the model against all past observations before planning, without any game-specific logic" |
| Memory across levels | Code | Observation history | implied |
| When to reset | not stated | | |
| When to call the LLM | Code | Called when model needs writing or refactoring | implied |

**Results**: "With GPT-5.5 high reasoning effort, the agent fully solved 15 games and achieved a mean per-game RHAE of 58.12%." (read, verbatim)

Quote: "The agent-facing prompts, workspace, and controller contain no game-specific code, game-specific prompts, hand-coded heuristics, hidden solutions, or other game-specific information." (read, verbatim)

---

### A10. Scaling Discovery through Test-Time Communication (arXiv 2609.21032)

**Source**: https://arxiv.org/abs/2609.21032
**Model**: Not specified (uses "coding agents")
**LLM calls per action**: Multiple agents in parallel
**Status**: read (HTML version)

| Decision | Made by | How (1 line) | Quote (verbatim) |
|----------|---------|--------------|-------------------|
| All individual decisions | LLM | Each agent operates independently | implied |
| Exploration strategy | Both | Breakthroughs shared through a directory across agents | "breakthroughs are shared through a directory" |
| Memory across levels | Code | Communication directory enables cross-agent memory | implied |

**Key ablation**: "a team of k communicating agents, team@k, matches the success rate of 4k independent agents, and this advantage grows with k" (read, verbatim)

"The effect is not merely efficiency: a task that no single agent can solve, a team of agents can solve reliably." (read, verbatim)

Communication ON vs OFF yields a 4× efficiency gain.

---

### A11. Preview Agents (ARC-AGI-3 Preview, Aug 2025)

#### Just Explore (3.64%)
**Source**: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore | **Model**: None (no LLM) | **Status**: read
Post-evaluation: "solved 12 out of 25 private levels... After correcting the graph-resetting bug post-evaluation... it solves a median of 17 private levels" (read)

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | Code | Segments frame into single-color connected components | "Segmenting the frame into single-color connected components... Detecting and masking likely status bars" |
| Object segmentation | Code | Groups segments into five priority tiers by button likelihood | "grouping segments into five priority tiers based on button likelihood (average size, salient color" |
| Tracking what changed | Code | Hashes masked images and tracks transition results | "For each action... tested flag, transition result, destination frame, and distance to the nearest frontier." |
| Choosing next action | Code | Selects untested actions from highest priority group | "Actions are taken from the highest-priority group with remaining untested actions" |
| Multi-step planning | Code | Graph explorer maintains paths to frontier frames | "From each known frame (graph node), the explorer maintains paths to frontier frames—those with untested actions" |
| Exploration strategy | Code | Brute-force level graph search | "A brute-force exploration method emerged and performed better for the public tasks." |

#### Blind Squirrel (6.71%, 2nd place)
**Source**: https://github.com/wd13ca/ARC-AGI-3-Agents | **Model**: ResNet18 (small CNN, trained during play; no LLM) | **Status**: read

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | Code | Stem converts grid cells into 16-dim embeddings | "The stem converts the grid's 16-possible cell values into a 16-dim embedding and places it through a conv layer before it feeds into the pre-trained backbone." |
| Object segmentation | Code | Segments grid into shapes as potential buttons | "I segmented the grid into shapes (i.e. groups of contiguous cells of the same colour) and treated each shape as a single potential 'button'." |
| Tracking what changed | Code | State graph stores history of all states and transitions | "The state graph stores a history of every previously encountered game state, the actions taken from that state, and the resulting next states." |
| Rule/world-model | Code | Assumes deterministic games; builds state graph | "I assumed that the games are deterministic: for a given state, a given action will always produce the same result." |
| Goal hypothesis | Code | Measures distance to level-winning state | "To train the model, we use the state graph to measure the distance (in terms of actions) between each state and the level-winning state." |
| Choosing next action | Code | Combines Action Value + Valid Actions models | "Values from the Action Value Model are combined with scores from the Valid Actions Model to determine the agent's next action." |
| Multi-step planning | Code | State graph avoids known game-over paths | "The state graph is used to identify and avoid actions that are known to lead to 'game over'." |
| Exploration strategy | Code | Valid Actions model down-weights unlikely actions | "a rules-based Valid Actions Model, that reduces the action space and down-weights actions that are unlikely to result in a grid change." |
| Memory across levels | Code | Model retrained after every other level | "The model is trained at the end of level 1, and then retrained at the end of every other level." |

#### Fluxonian (8.04%, Honorable Mention)
**Source**: https://github.com/FluxonApps/arc-prize | **Model**: GPT-4.1 (via OpenAI API) | **Status**: read

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | Code | DSL engine primitives handle pattern matching | "pattern_matcher.execute(context, operation='find_shapes', color=1, min_size=3)" |
| Object segmentation | Code | Primitives find shapes and colors | "Find shapes in the grid... Find colors and their distribution" |
| Tracking what changed | Code | DSL execution engine tracks history | "The DSL agent automatically: Learns common patterns from grid history" |
| Rule/world-model | LLM | Agent outputs DSL transformation rules as hypotheses | "Parse action and new rule from LLM response... Extracted DSL rule text (from LLM):" |
| Choosing next action | Both | DSL engine evaluates LLM-generated rules with confidence scores | "Match specific patterns and execute actions... ACTION1() confidence=0.9" |
| Multi-step planning | Code | Sequence rules handle multi-step actions | "Sequence Rules: Execute multiple actions in sequence" |
| Verifying hypothesis | Code | Agent tracks action success rates, adapts rule confidences | "The DSL agent automatically: Tracks action success rates... Adapts rule confidences based on performance" |
| Memory across levels | Code | Agent saves learned patterns | "Saves learned patterns for future use" |
| When to call LLM | Code | Called with current grid and DSL rules context | "user_content = self._build_user_message(current_grid, previous_grid, analysis, rules)" |

#### TOMAS Engine (3.70%)
**Source**: https://github.com/TesslaRay/tomas-engine-arc-agi-3 | **Model**: Google Gemini API (3 specialized agents) | **Status**: read | Described as "LLM — Limited Results, Crashed Often"

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | Both | AISTHESIS combines code object detection with LLM summaries | "The objective system found {object_count} distinct objects in the current environment." |
| Object segmentation | Code | Code bounds contiguous objects before passing to LLM | "Size: {obj.size} pixels, Bounds: rows {obj.bounds[0]}-{obj.bounds[1]}, cols {obj.bounds[2]}-{obj.bounds[3]}" |
| Tracking what changed | LLM | AISTHESIS LLM analyzes differences | "Analyze the CURRENT environment to help LOGOS understand what's available for interaction." |
| Rule/world-model | LLM | SOPHIA generates and tests hypotheses | "SOPHIA: Rapidly generates and tests game mechanic hypotheses" |
| Choosing next action | LLM | LOGOS uses simulated psychology | "LOGOS: Makes strategic decisions using human-like psychology... Chooses 1-5 action sequences based on confidence and emotional state" |
| Multi-step planning | LLM | LOGOS outputs multi-step sequences | "Chooses 1-5 action sequences based on confidence and emotional state" |
| Exploration strategy | LLM | Uses simulated frustration levels | "PSYCHOLOGICAL STATE: Exploring mode with moderate frustration (0.4)... Testing upward movement first to gather info" |
| Verifying hypothesis | LLM | LOGOS analyzes prediction accuracy | "Analyze previous prediction accuracy (when available): How well did your last expected_outcome match what AISTHESIS reported actually happened?" |
| Memory across levels | Code | Rule Consolidation stores successful patterns | "Rule Consolidation: 'Etches successful patterns into memory' when levels complete" |
| When to call LLM | Code | Architecture pipelines calls explicitly | "Game State → AISTHESIS → SOPHIA → LOGOS → Actions → Game State" |

Note: TOMAS was the most LLM-heavy Preview agent (nearly all decisions by LLM) and also one of the lowest-scoring (3.70%, 1 level completed in 79 actions).

#### GuidedRandomAgent (2.24%)
**Source**: https://github.com/GameDevGitHub/GuidedRandomAgent | **Model**: None (no LLM) | **Status**: read

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | Code | Analyzes pixel grid to identify distinct objects | "It analyzes the pixel grid each turn to identify distinct objects (like squares and rectangles)." |
| Object segmentation | Code | Identifies objects in pixel grid | "It analyzes the pixel grid each turn to identify distinct objects (like squares and rectangles)." |
| Tracking what changed | Code | Maintains memory of objects frame to frame | "It maintains a memory of these objects from one frame to the next." |
| Choosing next action | Code | Adjusts probabilities based on past action effectiveness | "It adjusts the probability of choosing each type based on whether past actions of that type resulted in a change to the game grid." |
| Exploration strategy | Code | Spontaneous Change Heuristic prioritizes objects that change | "Spontaneous Change Heuristic: code detects when an object changes... interprets this as a critical game event and applies a large, temporary 'buff' to the object's weight" |
| Memory across levels | Code | Checks step limits against previous levels | "Stops if level is too long. Also stops if prev level is too long b/c EX longer steps per level." |

#### Play Zero (4.37%)
**Source**: https://github.com/dhanaabhirajk/ARC-AGI-3-Agents | **Model**: Gemini 2.5 Pro / Flash | **LLM calls per action**: ~3 (1 Image LLM + 2 for goal check) | **Status**: read

| Decision | Made by | How | Quote |
|----------|---------|-----|-------|
| Perception | LLM | Image/Video LLMs analyze frames | "Takes actions using (1 Image LLM will current frame), goal achievent check (2 Images LLM call)." |
| Rule/world-model | Both | Uses LLMs for game analysis after goals achieved | "If goal achieved, do game analysis" |
| Goal hypothesis | LLM | Video/text LLMs define goals | "Do analysis and create goal with max actions limit (1 Video LLM call, 1 text LLM call)" |
| Choosing next action | Both | Alternates between random and LLM-chosen actions | "Takes actions using (1 Image LLM will current frame)" |
| Exploration strategy | Code | Random play exploring | "If max actions limit reached without achieving or level changed, do exploring (with random play)" |
| Verifying hypothesis | LLM | Image LLMs verify goal achievement | "goal achievent check (2 Images LLM call)." |
| When to call LLM | Code | Architecture orchestration logic | "1. Random play (do random probability after a threshold of actions met) 2. Do analysis and create goal" |

---

### A-Summary: Cross-Source Pattern in ARC-AGI-3

Across Twin, Tycho, OPINE-World, and the Executable World Models papers, a consistent architecture emerges:

| Component | Consistently done by | Notes |
|-----------|---------------------|-------|
| Writing the world model | LLM | All research agents agree |
| Verifying the world model against past observations | Code (automated replay) | All research agents agree — NEVER left to the LLM alone |
| Planning/pathfinding against a validated model | Code (BFS/search) | Twin, EWM agree; OPINE-World uses LLM |
| Choosing the exploration target | Code (heuristics/signals) | Twin, OPINE-World (ontology error), AERA (entropy budget) |
| Deciding WHEN to build a model | Varies — key differentiator | Tycho: LLM decides ("active abstraction"); Twin: code decides (counterexample triggers) |
| Per-step action selection | Code (after model is validated) | Twin; vs LLM in Kaggle winners |

The Kaggle Milestone #1 winners use a very different pattern: the LLM makes nearly ALL decisions, with minimal code infrastructure (harness, context management). The blog explicitly states "hand-crafted tools actually hurt the model."

---

## 3) Part B — General LLM-Agent Research on the LLM/Code Split

### B1. LLM-Modulo / "LLMs Can't Plan" (Kambhampati et al.)

**Source**: https://arxiv.org/abs/2402.01817 | **Status**: read
**Split**: LLM generates candidate plans; external model-based verifier checks hard constraints. The LLM is a "generator" and the planner/verifier is the "critic."
**Model**: GPT-4, GPT-3.5 (~175B+)
**Result**: "On average, only about 12% of the plans that the best LLM (GPT-4) generates are actually executable without errors and reach their goals." (read, verbatim)
**Ablation**: Performance deteriorates further when action/object names are obfuscated. Improves when subgoal interactions are removed by relaxing world models.

### B2. LLM+P

**Source**: https://arxiv.org/abs/2304.11477 | **Status**: snippet
**Split**: LLM translates natural-language problem → PDDL; classical PDDL planner finds solution; LLM translates back.
**Model**: GPT-4
**Result**: LLM+P solves **100%** of most evaluated domains; LLM alone solves **near 0%**.
**Ablation**: Removing external planner: 100% → near 0%.
**Quote**: "LLM+P takes in a natural language description of a planning problem, returns a correct plan in natural language, and bypasses the reasoning limitations of LLMs by delegating the actual planning to a classical planner." (snippet)

### B3. SayCan

**Source**: https://arxiv.org/abs/2204.01691 | **Status**: read
**Split**: LLM proposes high-level tasks ("Say"); learned value function scores physical feasibility ("Can").
**Model**: PaLM 540B
**Result**: 84% execution success rate on 101 instructions.
**Ablation (from paper)**:
- With affordance grounding (SayCan): **70% planning success rate**
- Without affordance grounding ("No VF"): **38% planning success rate**
**Quote**: "SayCan extracts the grounding from the robot's affordances, which are used to score the LLM's proposed actions." (snippet)

### B4. Voyager

**Source**: https://arxiv.org/abs/2305.16291 | **Status**: read
**Split**: LLM writes modular code skills; environment executes and provides feedback; verified skills stored in library.
**Model**: GPT-4
**Result**: 3.3× more unique items, 2.3× longer distances, 15.3× faster tech tree unlocks vs prior SOTA.
**Ablation (from paper)**:
- Without curriculum (random): discovered item count drops by **93%**
- Without skill library: agent plateaus in later stages
- Without self-verification: discovered item count drops by **73%**
- GPT-4 vs GPT-3.5: GPT-4 obtains **5.7× more unique items**
**Quote**: "Voyager is the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention." (snippet)

### B5. ReAct

**Source**: https://arxiv.org/abs/2210.03629 | **Status**: read
**Split**: LLM interleaves reasoning traces (thought) and actions at each step.
**Model**: PaLM 540B
**Ablation (from paper, AlfWorld)**:
- ReAct (Reason + Act): **71%** average success rate
- Act-only: **45%** average success rate
- (Reason-only/CoT evaluated on HotpotQA, not AlfWorld)
**Quote**: "ReAct prompts LLMs to generate both verbal reasoning traces and actions in an interleaved manner." (snippet)

### B6. Reflexion

**Source**: https://arxiv.org/abs/2303.11366 | **Status**: read
**Split**: LLM generates code/action; environment gives binary success/failure; LLM self-reflects with text feedback for next trial.
**Model**: GPT-4
**Ablation (from paper)**:
- AlfWorld with reflection: **97%** success rate vs **75%** without (base ReAct, saturated via retries)
- HumanEval with reflection: **91%** pass@1 vs **80%** without
**Quote**: "Reflexion is a framework that reinforces language agents not by updating weights, but through linguistic feedback." (snippet)

### B7. DECKARD

**Source**: https://arxiv.org/abs/2301.12050 | **Status**: snippet
**Split**: Alternates "Dream" phase (LLM hypothesizes Abstract World Model) and "Wake" phase (modular RL policies verify it).
**Model**: GPT-3 scale
**Result**: Orders of magnitude better sample efficiency than tabula rasa RL.
**Ablation**: Removing LLM Dream phase reduces to standard RL, severely slowing exploration.
**Quote**: "DECKARD operates by alternating between two phases to blend LLM-based knowledge with grounded environment experience." (snippet)

### B8. ELLM

**Source**: https://arxiv.org/abs/2302.06692 | **Status**: snippet
**Split**: LLM suggests goals based on state description; agent uses RL to achieve them.
**Model**: GPT-3, Codex
**Result**: Outperforms traditional exploration baselines in coverage (Crafter, Housekeep).
**Ablation**: Removing LLM goal suggestions → significantly worse coverage.
**Quote**: "ELLM guides agents to discover complex, common-sense behaviors more efficiently without requiring a human-in-the-loop during the training process." (snippet)

### B9. AdaPlanner

**Source**: https://arxiv.org/abs/2305.16653 | **Status**: snippet
**Split**: LLM generates complete plan; adaptively refines locally based on environmental feedback (no full replanning).
**Model**: GPT-3.5, GPT-4
**Result**: Outperforms ReAct by a large margin on AlfWorld.
**Ablation**: Full replanning from scratch → worse performance/latency than adaptive local refinement.
**Quote**: "AdaPlanner allows the LLM agent to adaptively refine its plan based on environmental feedback." (snippet)

### B10. Code as Policies

**Source**: https://arxiv.org/abs/2209.07753 | **Status**: read
**Split**: LLM writes Python code expressing spatial/logic policies that invoke robot control APIs directly.
**Model**: Codex (~12B), GPT-3 (175B)
**Ablation (from Table IV)**:
- Code Generation (LMP): **98%** mean success rate
- Chain-of-Thought Natural Language: **58%** mean success rate
- Vanilla Natural Language: **35%** mean success rate
**Quote**: "Code as Policies proposes using LLMs to write Python code that directly controls a robot." (snippet)

### B11. WorldCoder

**Source**: https://arxiv.org/abs/2402.12275 | **Status**: read
**Split**: LLM synthesizes Python code representing environment dynamics (world model); agent acts based on this programmatic model.
**Model**: GPT-4
**Ablation (from paper)**:
- WorldCoder: builds world model in **first 50 actions** (steps) on Sokoban
- Deep RL (PPO baseline): requires **>1,000,000 steps** for 2-box Sokoban
- Minigrid: PPO takes 10⁴–10⁵ steps; fails on UnlockPickup even after 300,000,000 steps
**Quote**: "WorldCoder constructs an explicit, interpretable world model by writing Python code." (snippet)

### B12. GIF-MCTS

**Source**: https://arxiv.org/abs/2405.15383 | **Status**: snippet
**Split**: MCTS guides LLM's generation, improvement, and fixing of executable code-based world models.
**Model**: GPT-4
**Result**: Surpassed all baselines across 18 RL environments (Code World Models Benchmark).
**Ablation**: Removing MCTS guidance → sharp drop in world model accuracy.
**Quote**: "GIF-MCTS uses MCTS to guide the LLM through the process of generating, improving, and fixing code." (snippet)

### B13. AutumnBench

**Source**: search results (exact arXiv ID unclear) | **Status**: snippet
**Split**: Benchmark: reward-free interaction phase → scored test phase.
**Model**: Gemini 2.5 Pro, Claude 4 Sonnet, OpenAI o3
**Result**: Humans significantly outperform frontier AI models across 129 tasks in 43 grid-world environments.
**Quote**: "AutumnBench uses a reward-free interaction phase followed by a scored test phase to measure world-building capabilities." (snippet)

---

### B-Summary: Measured Effects of Moving the LLM/Code Line

| Paper | What moved | Effect (numbers) |
|-------|-----------|-------------------|
| LLM-Modulo | LLM plans alone vs LLM+verifier | 12% valid plans (LLM alone) |
| LLM+P | LLM plans alone vs LLM+PDDL planner | ~0% → 100% |
| SayCan | LLM proposals alone vs LLM+affordance grounding | 38% → 70% planning success |
| Voyager | Without vs with skill library / curriculum / self-verification | 93% fewer items without curriculum; 73% fewer without self-verification; 5.7× GPT-4 vs GPT-3.5 |
| ReAct | Act-only vs Reason+Act | 45% → 71% (AlfWorld) |
| Reflexion | Without vs with self-reflection | 75% → 97% (AlfWorld); 80% → 91% (HumanEval) |
| Code as Policies | Natural language vs code generation | 35% → 98% success |
| WorldCoder | Deep RL vs LLM-written world model | 1M+ steps → 50 steps |
| Twin (ARC-AGI-3) | Direct play vs off-the-shelf Codex vs Twin harness | 7.8 → 61.1 → 93.3 RHAE |
| Coding Agents ablation | Textual vs executable (GPT-5.5) | Textual outperforms executable |

---

## 4) Part C — Small Models (≤ 8B and 27–32B): What They Do Reliably vs Unreliably

### Benchmark Numbers

| Claim | Model | Number | URL | Status |
|-------|-------|--------|-----|--------|
| **Constrained decoding (schema forcing)** drops reasoning accuracy by up to 27 points on reasoning tasks | Small models (0.6B–4B) | Up to 27 pt drop; DCCD recovers up to 24 pts | https://arxiv.org/abs/2603.03305 | snippet |
| **Spatial reasoning** collapses when grids are presented visually vs as adjacency lists | Frontier + small models | 80–86% (text/adjacency) → 16–34% (visual grids) | "Do LLMs Build Spatial World Models?" (2026) | snippet |
| **Counting** objects is a structural weakness; scaling model size does not fix it | General LLMs including GPT-4o, Gemini 1.5 | Not linearly scalable | CountBench / UNICBench studies | snippet |
| **Multi-step planning**: 8B vs 70B gap is large | Llama 3 8B vs 70B | GSM8K: 56–84% (8B) vs 76–95% (70B) | search results | snippet |
| **Multi-step planning**: 8B vs 70B on ARC-Challenge | Llama 3 8B vs 70B | ARC-Challenge: 78–83% (8B) vs 93–95% (70B) | search results | snippet |
| **JSON structural validity**: constrained decoding achieves 100%, naive prompting can give 0% | 1B–14B models | 0% naive → 100% structural validity (with constrained decoding) | https://arxiv.org/abs/2603.03305 | snippet |
| **StructEval**: structured output generation is harder than conversion; visual formats harder than text-only | o1-mini vs Llama-3-8B-Instruct | o1-mini: 75.58 avg; Llama-3-8B-Instruct: ~65.58 avg (~10 pts behind) | https://arxiv.org/abs/2505.20139 | read |

### Model-Specific Benchmarks

| Model | Size | MMLU / MMLU-Pro | GSM8K | HumanEval | IFEval | GPQA-Diamond | LiveCodeBench v6 | Source |
|-------|------|-----------------|-------|-----------|--------|--------------|------------------|--------|
| Qwen3-4B (kosa fine-tune) | 4B | ~65.76% | ~85.60% | not stated | not stated | not stated | not stated | snippet (Reddit) |
| Qwen3.5-4B | 4B | 79.1% (Pro) | not stated | not stated | 89.8% | 76.2% | not stated | snippet |
| Phi-4-mini | 3.8B | 67.3% | 88.6% | 74.4% | not stated | not stated | not stated | snippet |
| Gemma-4-31B | 31B | 85.2% (Pro) | not stated | not stated | not stated | 84.3% | 80.0% | snippet |
| Gemma-4-31B (agentic) | 31B | — | — | — | — | — | τ2-bench: 86.4% | snippet |

**Note**: There is no Gemma-4-27B; the Gemma-4 series has 26B MoE and 31B Dense variants.

### Key Observations (with citations)

**Qwen3.5-4B IFEval score (89.8%)**: This measures instruction-following including structured output. This is notably high for a 4B model, suggesting constrained-output tasks are feasible if the model was trained for them.

**Code generation at 4B is viable but limited**: Phi-4-mini gets 74.4% on HumanEval (3.8B); Qwen3-4B gets ~85.6% on GSM8K. These are adequate for simple code but likely insufficient for the complex multi-step program synthesis required for world modeling.

**The 27–32B tier is qualitatively different**: Gemma-4-31B scores 80.0% on LiveCodeBench v6 and 86.4% on τ2-bench (agentic tool use). These scores approach frontier-model capability for code and agentic tasks.

**Voyager ablation on model size**: GPT-4 obtains 5.7× more unique items than GPT-3.5 in Minecraft. (read, https://arxiv.org/abs/2305.16291). This measures the effect of model capability on the same LLM/code split.

**AERA with Qwen2.5-0.5B**: Even a 0.5B model achieved RHAE = 0.2116 (4/25 solved) when the exploration strategy was handled by code. (read, https://arxiv.org/abs/2605.25931)

---

## 5) Open Gaps

1. **Reki and forge detailed write-ups**: No repos, papers, or detailed Kaggle discussion posts found beyond the Milestone #1 blog. Exact scores for all three Milestone #1 winners are not publicly stated. The blog only says who won, not exact RHAE values.

2. **Tycho full results tables**: The HTML version of the Tycho paper was truncated before the results tables. The exact RHAE for always-build and never-build policies (needed to quantify the "active abstraction" ablation) was not extracted. Only the actor-requested delegation result (88.49 RHAE) was visible.

3. **OPINE-World ablation tables**: No specific ablation table comparing with/without ontology error guidance or with/without CEGIS was found in the extracted text. The mechanism is described but not individually ablated.

4. **Twin model-size comparison**: The Twin paper was read only with GPT-5.6 Sol. Whether they tested smaller models (and what the degradation curve looks like) was not found.

5. **LLM calls per action for research agents**: Most papers do not state the exact number of LLM calls per game action. Twin reports 224k tokens per action but not the call count.

6. **Kaggle Milestone #1 exact scores**: The blog post does not report exact RHAE values for Duck, Reki, or forge — only placement order and prize amounts.

7. **Small model spatial/grid benchmarks**: The "Do LLMs Build Spatial World Models?" paper was found only via snippet. The exact paper URL, model-by-model breakdown, and grid-size scaling curves were not obtained.

8. **Constrained decoding paper (arXiv:2603.03305)**: Read only via snippet. The exact model-by-model breakdown of accuracy drop from schema enforcement needs primary reading.

9. **AutumnBench exact arXiv ID**: The exact paper URL was not confirmed (listed as `2409.XXXX` in snippet).

10. **Qwen3.6-27B specific benchmarks**: The Kaggle winners use "Qwen 3.6 27B" but specific benchmarks for this model version (distinct from Qwen3-32B or Qwen3.5-4B) were not found.

11. **General "workflows vs agents" claim (Part B, item 14)**: The assertion that "workflows are superior for reliability while agents are reserved for exploratory subtasks" came from search snippets without a specific primary source URL. This needs verification.

12. **LLM+P, DECKARD, ELLM, AdaPlanner**: Read only via snippets. Full-text reading would provide more precise ablation numbers and experimental details.

13. **Memory across levels in ARC-AGI-3**: Most papers and agents do not explicitly discuss how information transfers between levels within the same game. Twin, OPINE-World, and the EWM paper are silent on this; only Tycho mentions "meta-reflection."
