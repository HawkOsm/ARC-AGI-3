# Where the line between LLM and code decisions goes (verified digest, 2026-09-26)

Collected by `agy` (Claude Opus 4.6; spec `.claude/scratch/research/llm_vs_algo/spec.md`; raw report
[`agy_llm_vs_algo_raw.md`](agy_llm_vs_algo_raw.md), 35 sources). Rows marked ✅ were re-read at the source by Claude
Code; the rest is agy's report and may be wrong. Two agy claims were corrected (see "Corrections").

## Verified evidence

| source | what was measured | ✅ quote / number |
|---|---|---|
| [Milestone #1 winners](https://arcprize.org/blog/arc-prize-2026-milestone-1) (Kaggle, 27–31B local models) | LLM picks the actions; little machinery | Tufa Labs: "hand-crafted tools actually hurt the model; letting it improvise worked better." forge: "The top-scoring run of this notebook used a profile that turns off all of the extra machinery." Reki: code keeps a "numpy click heuristic" (clicks prefer "small, rare-colored, button-like shapes") and a "Dead-signature" that "stops wasting clicks" on objects whose click never changes anything; LLM returns "a short plan, and the next 1-4 actions". No scores published. |
| [Twin](https://arxiv.org/abs/2608.14490) (frontier coding agent) | same model, direct play vs harness vs code-enforced verification + planning | "Played directly, the base model scores only 7.8%; an off-the-shelf harness increases it to 61.1%, whereas our twin world model increases the same base model to 93.3%". "The harness enforces that an action is not made until the program reproduces every previous observed game transition." |
| [Do Coding Agents Need Executable World Models?](https://arxiv.org/abs/2607.15439) | textual vs executable vs verified world model | "The complete verification treatment ranks first throughout, sometimes narrowly, but uses substantially more resources." "textual outperforms flexible-interface executable in both gpt-5.5 conditions." |
| [Explore Before You Solve (AERA)](https://arxiv.org/abs/2605.25931) | a 0.5B model inside a code-structured explore / verify / plan loop | "RHAE=0.2116 (4/25 solved)" with Qwen2.5-0.5B; "random and no-explore baselines score 0.0000". |
| [Draft-Conditioned Constrained Decoding](https://arxiv.org/abs/2603.03305) | forcing a schema on small models | constrained decoding "can distort generation when the model assigns low probability mass to valid continuations"; 1B on GSM8K: 15.2% constrained, 39.0% with "an unconstrained draft … generated first, and constrained decoding … then applied". |

Not re-verified (agy, general agent literature): LLM+P (LLM alone ~0% vs LLM + classical planner 100%), SayCan (38% →
70% with grounding), Code as Policies (35% natural language → 98% code), Voyager (73% fewer items without
self-verification), LLM-Modulo (~12% of GPT-4 plans executable). Direction consistent: an LLM proposes, code checks
and executes.

## Corrections to agy's report
- agy: "In all three cases, teams found that hand-crafted tools actually hurt". The blog says it of Tufa Labs only.
- agy: constrained decoding drops reasoning "up to 27 points"; not found at the source. Verified numbers above.
- agy: AERA "with exploration RHAE 0.5290": not in the paper's abstract/page; dropped.
- agy: Twin's harness "generates one candidate per progress signal": not in the abstract; unverified.

## Our own measurements (docs/tests/LOG.md)
- LLM chooses every action (drive mode): local 4B 2 levels vs brain 6; Kaggle 27B (hex screen) far worse than the brain.
- LLM picks the goal from the brain's list, code steers and checks (hypothesis mode): 11 levels vs brain 6 (one run).
