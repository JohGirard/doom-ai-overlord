# Laya — Research Notes

Research date: 2026-09-21. Package version analyzed: `laya 0.3.4` (installed locally in `.venv`),
`convaiinnovations/laya` on Hugging Face.

## What Laya is

Laya is a **non-autoregressive decision model**: a ModernBERT-large encoder (421M params total)
with a dedicated decision head trained with RLCD (REINFORCE-style learning against strictly
proper scoring rules). It never generates text — every question's options are scored at their own
`[MASK]` token and softmaxed per question. It is the **open-source (Apache 2.0) alternative to
TypeSafe's proprietary "Jev"** model — same decision-head family.

Answer space is defined **at request time** in the `questions` dict: no retraining needed for new
schemas. All questions in one `predict()` call are answered in a **single forward pass**
(~7 ms/question batched on GPU).

Three question primitives (`QTYPES = {"choice": 0, "score": 1, "noul": 2}`):

| Type | Criteria shape | Answer shape |
|---|---|---|
| `choice` | dict `{option_key: description}` (list auto-converted; `None` desc allowed) | `{"choice": str, "probabilities": {key: float}, "confidence": float}` |
| `score` | ordered list of level descriptions (0-indexed) | `{"score": float}` — **expected value** `Σ i·p(i)`, NOT an integer level |
| `noul` | optional `{"false": ..., "true": ...}` | `{"noul": float}` — calibrated `P(true)` in [0,1] |

Every answer also has `"action": {"act_probability": float}` (softmax of an auxiliary
act/escalate head) and the result dict has `usage.input_tokens`.

## Public API (verified against installed source)

```python
import laya

agent = laya.Agent("convaiinnovations/laya", device="cuda")  # or laya.load(...)
result = agent.predict(state, questions)   # predict == system_one
answers = result["answers"]
answers["qid"]["choice"]    # choice
answers["qid"]["score"]     # float, expected level index (0-based)
answers["qid"]["noul"]      # float, P(true)
```

Constructor args: `model_id_or_path` (HF repo id or local dir), `device` (None → cuda → mps → cpu),
`token` (falls back to `HF_TOKEN`), `subfolder` (`"multilingual"` / `"typed-decisions"` pick
checkpoints inside the bundle repo). Dtype: fp16 on CUDA (forced fp32 on CPU/MPS).

## Critical mechanics (this is where the old agent was wrong)

1. **There is NO backtick templating.** Instructions like `"What is in `targeting`?"` do not
   substitute the state value. The whole `state` dict is `json.dumps`-ed verbatim and appended
   after the question head (`common.py: build_sequence`). Backticked key names are only a
   *convention the model reads* — it must find the value itself in the raw JSON.
2. **Token budget is split.** Question head gets `head_max_len` (192 tokens English); the state
   gets the remainder of `max_len` (512) ≈ **320 tokens of JSON**. Keep the state dict small.
   Options are truncated to 48 tokens each; too many/long options raise
   `ValueError("options exceed head_max_len")`.
3. **`confidence` is normalized entropy (peakedness), not accuracy.** The shipped model stays
   confident while being wrong off-distribution (mean ECE 0.466 as shipped). Do not gate
   correctness on it.
4. **`score` (ordinal) is the weakest primitive** (SST-5 accuracy 0.372 zero-shot). Prefer
   `choice` and `noul`.
5. **The base checkpoint is near-chance on complex multi-step decisions zero-shot**
   (typed-decisions benchmark: 0.362 vs 0.318 random). It is a fast base to specialize, not a
   zero-shot reasoning engine. Questions must be **simple, well-specified, with distinct
   per-option descriptions** — ideally close to a literal mapping from state fields to choices.

## Consequences for the Doom agent

- Serialize a **compact** game state (≤ ~320 tokens of JSON): plain fields, short enum values.
- Ask for a **tactical choice whose options mirror the state's vocabulary literally**
  (`target is center` → `attack`), not free-text interpretation.
- Use `noul` for danger detection (reaction-speed modulation), not `score`.
- Add a **deterministic geometric fallback**: if the model's choice is absent from the action
  map, fall back to the geometry-derived tactic. Never let a misclassification mean "do nothing".
- Keep ammo/health guards deterministic (never attack with 0 ammo).

## Gotchas

- `questions={}` → `TypeError`; missing `type`/`instructions` → raw `KeyError` (no validation).
- If `laya.load()` hangs: transformers/TF abseil deadlock → run with `USE_TF=0`.
- English root checkpoint only; non-English state needs `subfolder="multilingual"`.
- Inference OOM on GPU → permanent silent migration to CPU mid-object (watch stderr).

## Sources

- Model card: https://huggingface.co/convaiinnovations/laya
- PyPI: https://pypi.org/project/laya/
- GitHub: https://github.com/NandhaKishorM (code on `research` branch)
- Comparison vs Jev: https://www.opensourcefactory.dev/2026/09/laya-vs-jev-4x-faster-17-points-behind.html
- Installed source: `.venv/Lib/site-packages/laya/` (`agent.py`, `common.py`, `presets.py`)
