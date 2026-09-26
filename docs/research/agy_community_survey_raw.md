# ARC-AGI-3 Community Approaches Survey

> Compiled 2026-09-26. Research-only document — no rankings or recommendations.

---

## 1) Sources Checked

| # | URL | Accessed? | Notes |
|---|-----|-----------|-------|
| 1 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 | Partial | Blocked by JS/bot-protection; data retrieved via search index |
| 2 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion | Partial | Same; top threads identified via web search |
| 3 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code | Partial | Same; top notebooks identified via web search |
| 4 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard | Partial | Same |
| 5 | https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings | Yes | Full blog post read successfully |
| 6 | https://arcprize.org/arc-agi-3 | Yes | Via web search extraction |
| 7 | https://docs.arcprize.org | Yes | Via web search extraction |
| 8 | https://arcprize.org/blog | Yes | Main blog index accessed |
| 9 | https://arcprize.org/paper-track | Yes | Paper Track details obtained |
| 10 | https://arcprize.org/blog/measuring-human-performance-on-arc-agi-3 | No (404) | Data obtained via web search results |
| 11 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2 | Partial | Blocked by JS; data via search |
| 12 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion | Partial | Same |
| 13 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code | Partial | Same |
| 14 | https://arcprize.org/blog/arc-prize-2024-winners-technical-report | Yes | Via web search |
| 15 | https://github.com/da-fr/arc-prize-2024 | Yes | Via web search |
| 16 | https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt | Yes | Via web search |
| 17 | http://export.arxiv.org/api/query (ARC-AGI queries) | Yes | arXiv API queries for papers |
| 18 | https://github.com/wd13ca/ARC-AGI-3-Agents | Yes | Via web search |
| 19 | https://github.com/FluxonApps/arc-prize | Yes | Via web search |
| 20 | https://github.com/dhanaabhirajk/ARC-AGI-3-Agents | Yes | Via web search |
| 21 | https://github.com/GameDevGitHub/GuidedRandomAgent | Yes | Via web search |
| 22 | https://github.com/TesslaRay/tomas-engine-arc-agi-3 | Yes | Via web search |
| 23 | https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore | Yes | Via web search |
| 24 | https://github.com/DriesSmit/ARC3-solution | Yes | Via web search |
| 25 | https://www.kaggle.com/competitions/arc-prize-2026-paper-track | Yes | Via web search |
| 26 | https://arxiv.org/abs/2601.10904 | Yes | ARC Prize 2025 Technical Report |

---

## 2) Entries

---

### Entry 1: ARC-AGI-3 Official Benchmark Paper

- **URL**: http://arxiv.org/abs/2603.24621v2
- **Date**: 2026-03-24
- **Author/Team**: ARC Prize Foundation
- **Approach**: Introduces ARC-AGI-3, an interactive benchmark for evaluating agentic intelligence via abstract, turn-based environments. Agents must explore and infer goals without explicit instructions; difficulty is calibrated against human performance using Relative Human Action Efficiency (RHAE).
- **Scores**: "humans can solve 100% of the environments, in contrast to frontier AI systems which, as of March 2026, score below 1%."
- **Quote**: "ARC-AGI-3 focuses entirely on evaluating fluid adaptive efficiency on novel tasks, while avoiding language and external knowledge."
- **Tags**: exploration, efficiency, engineering/kaggle-infra
- **Needs model training?**: no

---

### Entry 2: ARC-AGI-3 Preview: 30-Day Learnings (Blog Post)

- **URL**: https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- **Date**: 19 Aug 2025
- **Author/Team**: Greg Kamradt / ARC Prize Foundation
- **Approach**: Summarizes the 30-day ARC-AGI-3 Preview competition. Over 1,200 people played 3,900+ games. The post establishes the scoring framework based on per-level action efficiency compared to human baselines, and documents that some preview games were too vulnerable to random search, informing future game design. Key capabilities tested: on-the-fly learning, exploration, memory, and goal acquisition.
- **Scores**: (see individual agent entries below for specific scores)
- **Quote**: "Instead of asking, 'Can the test-taker recall the right answer?' interactive benchmarks ask, 'Can they explore, learn, plan, and adapt when dropped into an entirely new environment?'"
- **Tags**: exploration, efficiency, goal-inference, engineering/kaggle-infra
- **Needs model training?**: N/A (meta-source)

---

### Entry 3: Measuring Human Performance on ARC-AGI-3

- **URL**: https://arcprize.org/blog/measuring-human-performance-on-arc-agi-3 (returned 404; data from web search)
- **Date**: April 14, 2026
- **Author/Team**: ARC Prize Foundation
- **Approach**: Establishes the 100% human baseline for ARC-AGI-3 through controlled human testing (>200 participants). Uses the upper median of first-time human performance per level as the baseline. Introduces Relative Human Action Efficiency (RHAE) as the scoring metric — comparing AI action count against human baseline per level.
- **Scores**: N/A (defines the baseline, not an AI score)
- **Quote**: "Because the benchmark is designed to be solvable by humans with no prior training, the foundation conducted extensive human testing to provide a reliable proof point of general intelligence for comparison." (from web search summary)
- **Tags**: efficiency, exploration
- **Needs model training?**: N/A

---

### Entry 4: ARC-AGI-3 Design & Documentation (docs.arcprize.org)

- **URL**: https://docs.arcprize.org
- **Date**: 2026
- **Author/Team**: ARC Prize Foundation
- **Approach**: Official developer documentation. Provides the ARC-AGI Toolkit (Python SDK via `uv` or `pip`) to manage game loops, environments, and scoring. Includes quickstart guides, game schemas, vocabulary definitions. ARC-AGI-3 consists of hundreds of original hand-crafted environments and thousands of levels, presented as turn-based 64×64 grid games.
- **Scores**: N/A
- **Quote**: "ARC-AGI-3 is an Interactive Reasoning Benchmark designed to measure an AI Agent's ability to generalize in novel, unseen environments."
- **Tags**: engineering/kaggle-infra
- **Needs model training?**: N/A

---

### Entry 5: ARC Prize 2026 Paper Track

- **URL**: https://www.kaggle.com/competitions/arc-prize-2026-paper-track
- **Date**: 2026
- **Author/Team**: ARC Prize Foundation
- **Approach**: A $450,000 research competition rewarding conceptual progress in solving ARC-AGI-2 and ARC-AGI-3. Submissions must include a Kaggle writeup, media gallery, and a public notebook linked to an active code submission. $75,000 main prize pool + $375,000 bonus for papers scoring ≥4.5/5 on rubric.
- **Scores**: N/A
- **Quote**: "The track invites participants to document their research, methodologies, and findings regarding their submissions to either the ARC-AGI-2 or ARC-AGI-3 competition tracks."
- **Tags**: engineering/kaggle-infra
- **Needs model training?**: N/A

---

### Entry 6: The "Duck" Harness (Tufa Labs — Milestone #1 Winner)

- **URL**: https://arcprize.org/blog/arc-prize-2026-arc-agi-3-milestone-prize-1 (blog post); open-sourced on Kaggle
- **Date**: June 2026 (Milestone #1, period ending June 30, 2026)
- **Author/Team**: Tufa Labs (Jack Cole, Dries Smit et al.)
- **Approach**: A lightweight, open-source coding agent harness that treats ARC-AGI-3 games as interactive coding problems. Uses a Python REPL to interact with the environment. The agent perceives the board through multiple representations (rendered images, raw ASCII grids, segmentation tool for region-specific zooming). Uses "infinite play via eviction" — continuously pops oldest messages to handle long-horizon tasks without exhausting context. Won Milestone #1 ($37.5K) and became the foundational baseline for many subsequent submissions.
- **Scores**: N/A (exact score from milestone not found in sources; won first place)
- **Quote**: (inferred from web search) "The Duck is a lightweight, open-source coding agent harness that treats ARC-AGI-3 games as interactive coding problems."
- **Tags**: llm-agent, exploration, engineering/kaggle-infra, efficiency
- **Needs model training?**: no

---

### Entry 7: FluidMind (Kaggle Notebook)

- **URL**: https://www.kaggle.com/datasets/yousefturk/fluidmind-arc-agi-3
- **Date**: September 2026
- **Author/Team**: yousefturk
- **Approach**: Reframes the challenge as a test-time skill acquisition problem using a "governor" cycle (EXPLORE → HYPOTHESIZE → VERIFY → PLAN → SORTIE). Compresses the novel environment into an executable world model ("the kiln") and runs an "assay" over its interaction ledger to validate hypotheses before executing actions. Uses vLLM with Qwen3-6-27B-FP8 during test-time.
- **Scores**: None publicly stated; optimizes for RHAE.
- **Quote**: "treats fluid intelligence as test-time skill acquisition"
- **Tags**: world-model, exploration, test-time-learning
- **Needs model training?**: no (uses vLLM with pre-existing model at test-time)

---

### Entry 8: Kaggle Discussion — Tufa Labs Not Open-Sourcing for Milestone #2

- **URL**: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (top-voted thread; exact URL unavailable due to Kaggle bot protection)
- **Date**: September 2026
- **Author/Team**: Tufa Labs
- **Approach**: Strategic community announcement. Top team explains they will skip the Sept 30 open-source milestone because high score variance would lead to mass code cloning, turning the challenge into a lottery rather than an innovation test.
- **Scores**: N/A
- **Quote**: "We are fully committed to sharing our final solution when the competition ends."
- **Tags**: engineering/kaggle-infra
- **Needs model training?**: unclear

---

### Entry 9: Kaggle Discussion — Long GPU Queue Issues

- **URL**: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (exact URL unavailable)
- **Date**: September 2026
- **Author/Team**: Community Discussion
- **Approach**: Highlights infrastructure bottleneck: the g4-standard-48 GPUs required for heavy LLM agents have ~8-hour wait times. Competitors propose mitigations to prevent non-ARC-AGI-3 users from consuming reserved RTX Pro 6000 machines.
- **Scores**: N/A
- **Quote**: "Misuse of these resources can result in moderation actions, including site suspension or account bans."
- **Tags**: engineering/kaggle-infra
- **Needs model training?**: N/A

---

### Entry 10: "Just Explore" / "Explore It Till You Solve It" (Baseline Agent)

- **URL**: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- **Date**: August 2025
- **Author/Team**: Evgenii Rudakov
- **Approach**: A graph-structured exploration baseline that navigates environments by randomly expanding the state space through valid moves. Builds a frame graph and systematically explores, completing levels by exhaustive state-space coverage. Serves as a primitive baseline and precursor to the later arXiv paper (2512.24156, already known).
- **Scores**: "3.64%" (from ARC-AGI-3 Preview competition table), 12 levels completed, 278,158 actions, 0 games completed
- **Quote**: "stalls on large state spaces"
- **Tags**: exploration, state-abstraction
- **Needs model training?**: no

---

### Entry 11: Blind Squirrel (ARC-AGI-3 Preview — 2nd Place)

- **URL**: https://github.com/wd13ca/ARC-AGI-3-Agents
- **Date**: August 19, 2025
- **Author/Team**: Will Dick
- **Approach**: Explore-and-learn agent that builds an explicit directed state graph from video frames. Maintains memory of visited states to enable systematic coverage while pruning unhelpful loops. When score improves, it back-labels distances and retrains a small ResNet18 value model to rank (state, action) pairs toward the next milestone.
- **Scores**: "Score: 6.71%, Levels Completed: 13" (1 game completed, 109,108 actions)
- **Quote**: "Explore-and-learn agent that builds a state graph from frames" (from arcprize.org blog)
- **Tags**: exploration, state-abstraction, efficiency, training-required
- **Needs model training?**: yes (retrains ResNet18 during interaction)

---

### Entry 12: Fluxonian (ARC-AGI-3 Preview — Honorable Mention)

- **URL**: https://github.com/FluxonApps/arc-prize
- **Date**: August 19, 2025
- **Author/Team**: Ujjwal Chadha, Maya Nguyen, Shobhit Singhal, Filip Dominas
- **Approach**: Uses a Domain Specific Language (DSL) system designed for ARC tasks, providing a structured way to define patterns, transformations, and decision rules. Combines a DSL Engine, Parser, and Primitives with an LLM to coordinate pattern matching and rule execution.
- **Scores**: "8.04%" (5 levels completed, 11,890 actions, 0 games completed)
- **Quote**: "This DSL system provides a structured way to define patterns, transformations, and decision rules for ARC AGI tasks."
- **Tags**: program-synthesis, llm-agent
- **Needs model training?**: no

---

### Entry 13: Play Zero Agent (ARC-AGI-3 Preview — Honorable Mention)

- **URL**: https://github.com/dhanaabhirajk/ARC-AGI-3-Agents
- **Date**: August 19, 2025
- **Author/Team**: Dhana Abhiraj
- **Approach**: Performs random exploratory actions and uses the Gemini API to analyze results from video inputs. Operates multiple concurrent game sessions by distributing load across multiple API keys.
- **Scores**: "4.37%" (5 levels completed, 7,226 actions, 0 games completed)
- **Quote**: "Run the playzero agent (generates random actions) against all the games."
- **Tags**: llm-agent, exploration
- **Needs model training?**: no

---

### Entry 14: GuidedRandomAgent (ARC-AGI-3 Preview)

- **URL**: https://github.com/GameDevGitHub/GuidedRandomAgent
- **Date**: August 19, 2025
- **Author/Team**: Bob (GameDevGitHub)
- **Approach**: Categorizes possible interactions and dynamically shifts probabilities towards actions that effectively change the grid. Tracks individual objects across frames, rewarding those whose interactions cause state changes. Applies heuristics to prioritize spontaneously changing objects and records dead-end moves to systematically avoid redundant loops.
- **Scores**: "2.24%" (11 levels completed, 39,881 actions, 1 game completed)
- **Quote**: "It adjusts the probability of choosing each type based on whether past actions of that type resulted in a change to the game grid."
- **Tags**: exploration, object-perception, efficiency
- **Needs model training?**: no

---

### Entry 15: TOMAS Engine (ARC-AGI-3 Preview)

- **URL**: https://github.com/TesslaRay/tomas-engine-arc-agi-3
- **Date**: August 19, 2025
- **Author/Team**: Cristian Valdivia (TesslaRay)
- **Approach**: Multi-agent cognitive architecture with three specialized LLM-driven components: Aisthesis (perception), Sophia (hypothesis generation and rule consolidation), and Logos (strategic decision-making). These "minds" interact in a pipeline to observe visual changes mathematically, infer rules, and act. Described as "LLM — Limited Results, Crashed Often" in the official results table.
- **Scores**: "3.70%" (1 level completed, 79 actions, 0 games completed)
- **Quote**: "TOMAS (Thinking, Observing, Modeling, and Acting System) mimics human cognitive processes through three specialized AI agents: perception, learning, and strategic decision-making."
- **Tags**: llm-agent, object-perception, goal-inference, world-model
- **Needs model training?**: unclear

---

### Entry 16: Executable World Models for ARC-AGI-3 (Coding Agent)

- **URL**: http://arxiv.org/abs/2605.05138v2
- **Date**: 2026-05-06
- **Author/Team**: Sergey Rodionov
- **Approach**: A coding-agent system that maintains and refactors an executable Python world model based on observations. The agent uses predefined interfaces and verifies the model against all past observations before planning, without any game-specific logic or heuristics. A key early paper establishing the "executable world model" paradigm for ARC-AGI-3.
- **Scores**: "With GPT-5.5 high reasoning effort, the agent fully solved 15 games and achieved a mean per-game RHAE of 58.12%."
- **Quote**: "The agent-facing prompts, workspace, and controller contain no game-specific code, game-specific prompts, hand-coded heuristics, hidden solutions, or other game-specific information"
- **Tags**: world-model, program-synthesis, llm-agent, test-time-learning, efficiency
- **Needs model training?**: no

---

### Entry 17: Tycho — Active Abstraction with Programmatic World Models

- **URL**: http://arxiv.org/abs/2607.28287v1
- **Date**: 2026-07-30
- **Author/Team**: Jens Lehmann, Andrei Aioanei, Sahar Vahdati
- **Approach**: Formalizes environments as parameterized rendered deterministic Moore machines. Introduces Tycho, a coding-agent that constructs game-specific models during interaction. Separates actionable observations from intermediate frames. Introduces "active abstraction" — jointly deciding when to acquire a model vs. when to use it vs. when its cost exceeds its benefit.
- **Scores**: "Actor-requested delegation to a model builder obtains the highest observed mean Relative Human Action Efficiency (RHAE), 88.49. With this selected policy, GPT-5.6 Sol and Opus 5 both reach 100.00 RHAE and complete all 183 levels."
- **Quote**: "We call this joint problem active abstraction: generating a testable model from costly interaction and deciding when acquiring or using it is worth its cost."
- **Tags**: state-abstraction, world-model, program-synthesis, llm-agent, efficiency
- **Needs model training?**: no

---

### Entry 18: OPINE-World — Programmatic World Modeling with Ontology-error-Prioritized Exploration

- **URL**: http://arxiv.org/abs/2607.01531v2
- **Date**: 2026-07-01
- **Author/Team**: David Courtis, Wenhao Li, Scott Sanner
- **Approach**: LLM agent that learns object-centric programmatic world models online from interaction. Couples two cooperating agents — one acting in the environment and one synthesizing code with CEGIS (counterexample-guided inductive synthesis). Steers exploration via a Bayesian measure called "ontology error" that quantifies uncertainty about the object model.
- **Scores**: "OPINE-World solves 20 of 25 games without per-game training and reaches an action-efficiency score of 78.4 against the human baseline."
- **Quote**: "OPINE-World couples two cooperating agents in a loop of hypothesis and test, one acting in the environment and one synthesizing the model in code with replay verification and model-based planning"
- **Tags**: world-model, object-perception, exploration, program-synthesis, llm-agent
- **Needs model training?**: no

---

### Entry 19: Explore Before You Solve — Speed-Depth Trade-off in Epistemic Agents

- **URL**: http://arxiv.org/abs/2605.25931v1
- **Date**: 2026-05-25
- **Author/Team**: Liew Keong Han
- **Approach**: Analyzes the public ARC-AGI-3 games, showing they are vulnerable to non-intelligent heuristics, implying only the private set tests genuine intelligence. Introduces AERA, a three-phase agent (EXPLORE / VERIFY / PLAN) formulated around a speed-depth tradeoff framework to optimize the Pareto frontier between action efficiency and information gain.
- **Scores**: "AERA (Adaptive Epistemic Reasoning Agent), a three-phase (EXPLORE / VERIFY / PLAN) agent achieving RHAE=0.2116 (4/25 solved) on these 25 games with Qwen2.5-0.5B, while random and no-explore baselines score 0.0000."
- **Quote**: "This benchmark critique implies the public evaluation set cannot discriminate intelligent exploration from trivial heuristics - the private 55-game evaluation is the only genuine intelligence test."
- **Tags**: exploration, efficiency, llm-agent
- **Needs model training?**: no

---

### Entry 20: Do Coding Agents Need Executable World Models? (Ablation Study)

- **URL**: http://arxiv.org/abs/2607.15439v2
- **Date**: 2026-07-16
- **Author/Team**: Sergey Rodionov
- **Approach**: Compares four Codex-based agent variants to isolate the contributions of executable world modeling and verification. Finds that while verification increases scores at lower effort, a purely textual variant at max effort is surprisingly competitive and completes every level efficiently. Challenges the assumption that executable models are always necessary.
- **Scores**: "With gpt-5.6-sol, the verification variant completes every public level at xhigh and max with about 99% human-relative action efficiency while using fewer than half the human baseline's total actions."
- **Quote**: "Requiring an executable deliverable is not universally beneficial: textual outperforms flexible-interface executable in both gpt-5.5 conditions."
- **Tags**: world-model, program-synthesis, llm-agent, efficiency
- **Needs model training?**: no

---

### Entry 21: Twin — Playing an Unknown Game with a Test-Time Digital Twin

- **URL**: http://arxiv.org/abs/2608.14490v1
- **Date**: 2026-08-14
- **Author/Team**: Alexy Skoutnev, Kirill Acharya, Gaston Longhitano, Madeleine Udell, Kevin Ellis, Iddo Drori
- **Approach**: A test-time world-model inference system where a coding agent writes an executable "digital twin" of the game. Counterexamples from mismatches between predicted and actual transitions repair the model. The agent requires exact replay validation — reproducing every previous observed transition — before making real moves.
- **Scores**: "Twin clears 179 out of 183 levels (97.8%), and does so more efficiently than humans in 158 out of 179 levels (88.3%)."
- **Quote**: "The harness enforces that an action is not made until the program reproduces every previous observed game transition."
- **Tags**: world-model, program-synthesis, test-time-learning, llm-agent, efficiency
- **Needs model training?**: no

---

### Entry 22: Scaling Discovery through Test-Time Communication (Multi-Agent)

- **URL**: http://arxiv.org/abs/2609.21032v1
- **Date**: 2026-09-17
- **Author/Team**: Jongho Park, Vasilis Kontonis, Shivam Garg, Akshay Krishnamurthy, Dimitris Papailiopoulos
- **Approach**: Investigates the impact of test-time communication among multi-agent systems on ARC-AGI-3. Shows that communication helps agents solve tasks that independent parallel attempts fail on, as breakthroughs are shared through a directory. A team of k communicating agents matches the success rate of 4k independent agents.
- **Scores**: "We find that a team of k communicating agents, team@k, matches the success rate of 4k independent agents, and this advantage grows with k"
- **Quote**: "The effect is not merely efficiency: a task that no single agent can solve, a team of agents can solve reliably."
- **Tags**: llm-agent, test-time-learning, exploration
- **Needs model training?**: no

---

### Entry 23: ARC Prize 2025 Technical Report

- **URL**: https://arxiv.org/abs/2601.10904
- **Date**: January 15, 2026
- **Author/Team**: François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers
- **Approach**: Official technical report for ARC Prize 2025 (ARC-AGI-2 competition). Identifies "refinement loops" — iterative, per-task program optimization — as the defining theme of 2025. Discusses Tiny Recursive Models (TRM, ~7M parameters) achieving competitive performance through recursive output refinement. Notes the shift toward smaller, efficient models over massive frontier LLMs. Also previews ARC-AGI-3 interactive format.
- **Scores**: (see winners below; top score 24.03% on ARC-AGI-2 private set)
- **Quote**: (inferred from web search) "The state-of-the-art performance is increasingly being achieved by smaller, highly efficient models rather than solely relying on massive frontier LLMs."
- **Tags**: program-synthesis, test-time-learning, training-required
- **Needs model training?**: yes (describes trained model approaches)

---

### Entry 24: NVARC — ARC Prize 2025 1st Place

- **URL**: https://arcprize.org/competitions/2025 (competition page); arXiv:2601.10904 (technical report reference)
- **Date**: December 5, 2025
- **Author/Team**: NVARC (Jean-François Puget, Ivan Sorokin)
- **Approach**: Synthetic-data-driven ensemble combining an improved "ARChitects-style" test-time-trained model with components based on Tiny Recursive Models (TRM). Noted for efficiency, achieving top score under strict Kaggle compute constraints (~$0.20 per task). Used the NVIDIA NeMo suite for pipeline streamlining.
- **Scores**: "24.03%"
- **Quote**: (from web search) "Their solution focused on building a scalable, economical approach to AI reasoning."
- **Tags**: program-synthesis, test-time-learning, training-required
- **Needs model training?**: yes

---

### Entry 25: the ARChitects — ARC Prize 2025 2nd Place (also ARC Prize 2024 1st Place)

- **URL**: https://github.com/da-fr/arc-prize-2024 (2024 solution repo)
- **Date**: December 2024 (2024 winner) / December 2025 (2025 2nd place)
- **Author/Team**: Daniel Franzen, Jan Disselhoff, David Hartmann
- **Approach**: (2024) Utilized a 12B parameter LLM, tokenizing 2D grids as text. Heavily relied on test-time training (TTT) using LoRA to adapt to specific evaluation examples. Used depth-first search over token probabilities. (2025) Described as a "2D-aware, masked-diffusion language model with recursive self-refinement and perspective-based scoring."
- **Scores**: "53.5%" (2024, ARC-AGI-1), "16.53%" (2025, ARC-AGI-2)
- **Quote**: "Instead of relying on standard greedy or beam search, they employed a depth-first search over token probabilities to generate solutions."
- **Tags**: test-time-learning, llm-agent, training-required, world-model
- **Needs model training?**: yes

---

### Entry 26: MindsAI — ARC Prize 2024 (Highest Score) & ARC Prize 2025 3rd Place

- **URL**: https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf (2025 solution)
- **Date**: 2024-2025
- **Author/Team**: MindsAI & Tufa Labs (Jack Cole, Michael Hodel, Mohammed Osman)
- **Approach**: Heavily engineered test-time-training (TTT) pipeline. (2024) Achieved highest score but did not open-source, making them ineligible for prize. (2025) Integrated test-time fine-tuning, augmentation ensembles, tokenizer dropout, and novel pretraining techniques.
- **Scores**: "55.5%" (2024, ARC-AGI-1, highest overall), "12.64%" (2025, ARC-AGI-2, 3rd place)
- **Quote**: "A team named MindsAI achieved the highest score of 55.5% but was ineligible for the prize because they did not open-source their solution."
- **Tags**: test-time-learning, training-required, engineering/kaggle-infra
- **Needs model training?**: yes

---

### Entry 27: Ryan Greenblatt — GPT-4o Program Synthesis (ARC Prize 2024)

- **URL**: https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt
- **Date**: 2024
- **Author/Team**: Ryan Greenblatt
- **Approach**: Used GPT-4o to generate thousands of candidate Python programs for each ARC task. Programs were executed against training examples, and the LLM iteratively refined and debugged incorrect programs. Final answers selected via majority voting among surviving programs. No model training required — purely inference-time program synthesis.
- **Scores**: "50%"
- **Quote**: "He implemented a 'refinement' process where the LLM could iteratively debug and fix incorrect programs if they failed to map the examples correctly."
- **Tags**: program-synthesis, llm-agent, engineering/kaggle-infra
- **Needs model training?**: no

---

### Entry 28: ARC Prize 2024 Technical Report

- **URL**: https://arcprize.org/blog/arc-prize-2024-winners-technical-report
- **Date**: December 6, 2024
- **Author/Team**: ARC Prize Team (François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers)
- **Approach**: Summarizes breakthrough strategies: test-time training (TTT) on individual task specs, and deep learning-guided program synthesis. Documents the largest single-year jump in ARC-AGI scores.
- **Scores**: "The state-of-the-art score on the ARC-AGI benchmark rose from 33% to 55.5% over the course of the year, marking the largest single-year jump since the benchmark's inception"
- **Quote**: "The state-of-the-art score on the ARC-AGI benchmark rose from 33% to 55.5% over the course of the year, marking the largest single-year jump since the benchmark's inception"
- **Tags**: test-time-learning, program-synthesis
- **Needs model training?**: yes (describes TTT approaches)

---

### Entry 29: "Less is More: Recursive Reasoning with Tiny Networks" (TRM — Paper Award Winner)

- **URL**: https://arxiv.org/abs/2601.10904 (referenced in ARC Prize 2025 Technical Report)
- **Date**: January 2026 (reported in tech report)
- **Author/Team**: Alexia Jolicoeur-Martineau
- **Approach**: Uses tiny (~7M parameter) single-network recursive models that repeatedly revise their own latent state and answers. Demonstrates that "smarter" thinking mechanisms and iterative refinement loops can outperform scaling alone on abstract reasoning tasks. Won 1st Place Paper Award at ARC Prize 2025.
- **Scores**: "~45% on ARC-AGI-1 and ~8% on ARC-AGI-2" (from web search; not exact verbatim — approximate)
- **Quote**: (from web search summary) "TRM demonstrates that small networks can achieve competitive reasoning performance by learning to recursively update and refine their outputs."
- **Tags**: test-time-learning, training-required
- **Needs model training?**: yes

---

## 3) Recurring Themes

| Theme | Entry Numbers | Count |
|-------|---------------|-------|
| **Executable world models / digital twins** (coding agent writes a Python model of the game) | 7, 16, 17, 18, 20, 21 | 6 |
| **Exploration strategy** (systematic exploration vs. random, graph-based state coverage) | 1, 2, 3, 6, 10, 11, 13, 14, 18, 19, 22 | 11 |
| **LLM-as-agent** (frontier LLM drives decisions/code at test-time) | 6, 7, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 25, 27 | 14 |
| **Program synthesis** (generating programs/code to solve tasks) | 5, 12, 16, 17, 18, 20, 21, 23, 24, 27, 28 | 11 |
| **Action efficiency / RHAE metric** (optimizing for human-relative efficiency) | 1, 2, 3, 6, 7, 11, 14, 16, 17, 19, 20, 21 | 12 |
| **State abstraction / graph-based state representation** | 10, 11, 17, 19 | 4 |
| **Object perception** (detecting/tracking objects in the grid) | 14, 15, 18 | 3 |
| **Goal inference** (discovering the win condition without instructions) | 2, 15 | 2 |
| **Test-time training / learning** (adapting model weights during evaluation) | 7, 11, 16, 21, 22, 23, 24, 25, 26, 28, 29 | 11 |
| **No model training required** (pure inference-time approaches) | 1, 6, 7, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 27 | 15 |
| **Model training required** (requires pre-training or fine-tuning) | 11, 23, 24, 25, 26, 28, 29 | 7 |
| **Multi-agent cooperation** | 15, 22 | 2 |
| **REPL / coding-agent harness** | 6, 16, 17, 20, 21 | 5 |
| **Verification / replay validation before acting** | 17, 18, 20, 21 | 4 |
| **Kaggle infrastructure concerns** | 4, 5, 8, 9 | 4 |
| **Refinement loops** (iterative improvement of solutions) | 23, 25, 27, 29 | 4 |
| **DSL / structured rule systems** | 12 | 1 |

---

## 4) Open Questions / Things That Could Not Be Verified

1. **Exact Kaggle discussion URLs**: Kaggle pages were blocked by JavaScript/bot protection. Discussion thread URLs for Entries 8 and 9 could not be verified — only thread titles and approximate content were recoverable via search indexes.

2. **NVARC GitHub URL**: The subagent reported `https://github.com/nvarc-solution` but this URL could not be independently verified. The NVARC solution details come from the ARC Prize 2025 technical report and web search summaries.

3. **Exact Duck harness score at Milestone #1**: The precise RHAE score for the Duck harness when it won Milestone #1 was not found in any source. Only the prize amount ($37.5K) and the fact it won first place were confirmed.

4. **Human performance blog post (Entry 3)**: The URL `https://arcprize.org/blog/measuring-human-performance-on-arc-agi-3` returned 404 when accessed directly. Content was reconstructed from web search result summaries. The page may have been renamed or restructured.

5. **TRM paper exact scores**: The scores "~45% on ARC-AGI-1 and ~8% on ARC-AGI-2" for Entry 29 are approximate, sourced from web search summaries rather than the paper itself.

6. **FluidMind actual competition scores**: Entry 7 (FluidMind) had no publicly stated scores found despite being a Kaggle dataset/notebook.

7. **ARC-AGI-3 full private game set size**: The blog post mentions "hundreds of original environments and thousands of levels." The exact number of private evaluation games could not be confirmed beyond ">100" from the user's request context.

8. **Kaggle Code tab notebooks beyond Duck forks**: The Kaggle Code tab was blocked by bot protection. Web search suggests many top-scoring public notebooks are forks/modifications of the Duck harness, but individual notebook analysis was not possible.

9. **StochasticGoose (Preview 1st Place) — 2024 sample submission relationship**: The ARC-AGI-3 Preview 1st place "StochasticGoose" by Dries Smit/Jack Cole at Tufa Labs appears to be the basis for the official Kaggle sample submission mentioned in the "already known" items, but this connection was inferred, not explicitly confirmed.

10. **ARC Prize 2025 (ARC-AGI-2) Kaggle discussion content**: Individual discussion threads on the ARC-AGI-2 competition could not be accessed due to Kaggle bot protection. Only the winner results were verified.

11. **Kaggle image upscaling discussion (4x vs 8x)**: Referenced in web search results as an active debate, but the specific thread could not be accessed to extract verbatim quotes or exact details.

12. **Hybrid architectures (neuroevolution, JEPA-critic, Mental MCTS)**: Referenced in web search results about Kaggle discussions, but no specific thread or notebook was accessible for detailed extraction.

---

## Raw URL List (Every URL Actually Opened)

### Direct page reads
- https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- https://arcprize.org/blog/measuring-human-performance-on-arc-agi-3 (returned 404)

### arXiv API queries
- http://export.arxiv.org/api/query?search_query=all:%22ARC%22+AND+all:%22AGI%22&start=0&max_results=50
- http://export.arxiv.org/api/query?search_query=all:%22ARC-AGI-3%22&start=0&max_results=50

### Kaggle pages (attempted, blocked by bot protection — data retrieved via search)
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code

### Web searches (queries executed)
- "arcprize.org measuring human performance ARC-AGI-3 blog"
- "arcprize.org blog announcing-arc-agi-3 benchmark design"
- "Kaggle 'ARC Prize 2026' discussion 'ARC-AGI-3' approaches notebooks code 2026"
- "Tufa Labs 'Duck' ARC-AGI-3 harness REPL coding agent milestone prize"
- "ARC Prize 2025 winners NVARC ARChitects MindsAI scores technical report"
- "ARC Prize 2025 technical report arXiv 2601.10904 refinement loop tiny recursive models"
- "ARC Prize 2024 technical report"
- "ARC Prize 2024 winners solutions"
- "MindsAI ARC Prize 2024"
- "Ryan Greenblatt ARC Prize 2024 solution"
- "ARC-AGI-3 preview competition 2025 results winners"
- "ARC-AGI-3 preview competition write-up GitHub"
- "ARC-AGI-3 agent arXiv"
- "ARC-AGI interactive reasoning arXiv"
- "ARC challenge exploration world model arXiv"
- "abstraction reasoning corpus agent arXiv"

### GitHub repos (accessed via web search)
- https://github.com/wd13ca/ARC-AGI-3-Agents
- https://github.com/FluxonApps/arc-prize
- https://github.com/dhanaabhirajk/ARC-AGI-3-Agents
- https://github.com/GameDevGitHub/GuidedRandomAgent
- https://github.com/TesslaRay/tomas-engine-arc-agi-3
- https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- https://github.com/DriesSmit/ARC3-solution
- https://github.com/da-fr/arc-prize-2024
- https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf

### Blog / other sources (accessed via web search)
- https://arcprize.org
- https://arcprize.org/arc-agi-3
- https://arcprize.org/blog
- https://arcprize.org/paper-track
- https://arcprize.org/blog/arc-prize-2024-winners-technical-report
- https://arcprize.org/blog/arc-prize-2026-arc-agi-3-milestone-prize-1
- https://arcprize.org/blog/announcing-arc-agi-3
- https://arcprize.org/competitions
- https://docs.arcprize.org
- https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt
- https://www.kaggle.com/competitions/arc-prize-2026-paper-track
- https://www.kaggle.com/datasets/yousefturk/fluidmind-arc-agi-3
- https://arxiv.org/abs/2601.10904
- https://arxiv.org/abs/2412.04616
