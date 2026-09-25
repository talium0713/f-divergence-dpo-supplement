# AlpacaEval protocol

How we run AlpacaEval in this project, so a second person can produce numbers that pool with ours.
Two stages: **generate on a GPU (offline)**, then **judge where there is internet**. The compute node
has no outbound network, so the two cannot be merged.

Scripts: `llm/gen_alpaca.py` (stage 1), `llm/run_alpaca_lc.sh` and `llm/run_alpaca_judge.sh` (stage 2).

---

## Stage 1 — generation

```bash
python gen_alpaca.py \
  --model     <policy dir>                       # merged policy, not an adapter
  --prompts   data/alpaca_eval.jsonl             # 805 prompts
  --generator <tag>                              # name that appears in the output and on the leaderboard
  --max-new   512 \
  --out       jobs/alpaca_lc/<tag>.json
```

Settings that must not change, or the runs stop being comparable:

| item | value |
|---|---|
| Prompts | AlpacaEval 2.0, all 805 |
| Decoding | **greedy** (`do_sample=False`), no temperature, no repetition penalty |
| `max_new_tokens` | **512** — the arXiv:2512.00778 protocol |
| Prompt format | the tokenizer's own chat template, `add_generation_prompt=True` |
| Qwen3 thinking | **off** (`enable_thinking=False`), so no `<think>` block is generated |
| Precision | bf16, single GPU, no quantization |
| Special tokens | the template already inserts them (`add_special_tokens=False` at tokenization) |

Note the difference from our Arena-Hard runs, which use `max_new 1024` and `repetition_penalty 1.05`.
AlpacaEval here uses neither. Do not mix the two settings inside one table.

Output is AlpacaEval's own JSON: a list of `{"instruction", "output", "generator"}`. One 4B LoRA
policy over 805 prompts takes about 80 minutes on one L40S.

## Stage 2 — judging

Run on a cluster **login** node (`klogin*`, has internet) or on a laptop. Never under `sbatch`.

```bash
export OPENAI_API_KEY=sk-...          # or put the key in llm/openai_key.txt
bash run_alpaca_lc.sh jobs/alpaca_lc/<tag>.json [more .json ...]
```

`run_alpaca_lc.sh` is the one to use for anything that goes in the paper: it judges against the
**bundled gpt-4-turbo reference** with the `weighted_alpaca_eval_gpt4_turbo` annotator, which is the
standard AlpacaEval 2.0 setup and reports both

- **LC** — length-controlled win rate (the headline number), and
- **WR** — raw win rate.

It prints `-> LC = ...  WR = ...` per model and writes the full output under
`results/bench/ae_lc_<tag>/`.

`run_alpaca_judge.sh` exists too but answers a different question: it compares a candidate against
**our own SFT baseline** rather than the standard reference. Its numbers are not comparable to
published AlpacaEval 2.0 tables. Use it only if that is what you want.

---

## Naming

`alpaca_<block>_<div>_<norm>_s<seed>`, e.g. `alpaca_lora4b_kl_amari_s1`.
Code keys: **`kl` = Reverse KL, `rkl` = Forward KL** (this trips people up), plus `adiv`, `js`, `hel`,
`chi2`. Norm is `amari` or `canon`. Keeping the tag consistent lets both arms of a pair be matched
automatically later.

## Checks before paying for a judge

Judging costs API credit, so look at the generations first.

```python
import json, re
d = json.load(open("jobs/alpaca_lc/<tag>.json"))
loop = re.compile(r"(.{4,}?)\1{9,}", re.S)        # a short unit repeated >= 10 times
bad = [x for x in d if loop.search(x["output"])]
print(len(d), "responses,", len(bad), "collapsed", f"({100*len(bad)/len(d):.1f}%)")
```

Expect 805 responses and no empty outputs. A few percent of repetition is normal for the Amari arms;
a large fraction means something is wrong with the checkpoint or the decoding settings.

**This is not hypothetical.** On 2026-09-24 the first run, `alpaca_lora4b_kl_amari_s1` (4B LoRA,
Reverse KL, Amari form, seed 1), came back with **241 of 805 responses (29.9 %) collapsed into
repetition** — units like `ачи`, `öffuser`, `czasownik` repeated to the token limit, different
garbage per prompt. The remaining 564 are fluent and on topic. The same checkpoint shows only 1.6 %
repetition on Arena-Hard at `max_new 1024` with `repetition_penalty 1.05`, so the AlpacaEval setting
(no penalty, 512 tokens) exposes far more of it. Judge the paired canonical arm before concluding
anything: if canonical collapses at a similar rate the cause is the decoding setup, and if it stays
near zero the gap is a real property of the Amari form.

## What to report

LC as the headline, WR beside it, and the number of prompts that survived. State the decoding
settings (greedy, 512 tokens, no repetition penalty) next to the number — with collapse rates this
sensitive to the cap, a bare win rate is not reproducible.
