"""Every judged canonical-vs-Amari head-to-head, as one matrix per block.

Run from llm/results/bench/arena_v01/ (it globs h2h_*/ there):
    cd results/bench/arena_v01 && python ../../../master_table.py
The Notion tracker's master table is this output; re-run it after any new judgement rather than
editing the numbers by hand.

Reads results/bench/arena_v01/h2h_*/*.json (the summaries pairwise_h2h.py writes) and reports the
CANONICAL win rate, i.e. 100 - the Amari win rate the file stores. Where a pair was judged twice the
later/at-1910 file wins; the discarded ones are listed at the end.
"""
import json, glob, re, os

DIVS = ["kl", "rkl", "adiv", "js", "hel", "chi2"]          # code keys: kl = RKL, rkl = FKL
NICE = {"kl": "Reverse KL", "rkl": "Forward KL", "adiv": "alpha-div (0.5)", "js": "Jensen-Shannon",
        "hel": "Sq. Hellinger", "chi2": "Pearson chi2"}
BLOCK = {"lora1p7b": "1.7B LoRA", "lora4b": "4B LoRA", "lora8b": "8B LoRA",
         "lorallama1b": "Llama-1B LoRA", "scale4b": "4B full-FT", "scale8b": "8B full-FT",
         None: "1.7B full-FT"}
ORDER = ["1.7B full-FT", "4B full-FT", "8B full-FT", "1.7B LoRA", "4B LoRA", "8B LoRA",
         "Llama-1B LoRA", "Gemma-1B LoRA"]
cells, used = {}, {}


def put(block, div, seed, path, pref=1):
    d = json.load(open(path))
    key = (block, div, seed)
    if key in cells and used[key][1] >= pref:
        return
    cells[key] = (100 - d["win_rate"], 100 - d["ci_hi"], 100 - d["ci_lo"], d["n_games"])
    used[key] = (path, pref)


pat = re.compile(r"h2h_(?:(lora1p7b|lora4b|lora8b|lorallama1b|scale4b|scale8b)_)?"
                 r"(?:(kl|rkl|fkl|adiv|js|hel|chi2)_)?amari_vs_canon(?:_s(\d))?(.*)\.json$")
files = [f for f in glob.glob("h2h_*/*.json") if "_raw" not in f and "vs_base" not in f]
for f in files:
    base = os.path.basename(f)
    if "loragemma1b" in base:
        put("Gemma-1B LoRA", "kl", "0", f); continue
    m = pat.match(base)
    if not m:
        continue
    blk, div, seed, suf = m.group(1), m.group(2) or "kl", m.group(3) or "0", m.group(4)
    div = "rkl" if div == "fkl" else div      # the 1.7B seed-0 files spell Forward KL "fkl"
    block = BLOCK[blk]
    pref = 2 if "1910" in suf else (0 if (block == "4B LoRA" and "mini2" in suf) else 1)
    put(block, div, seed, f, pref)

for blk in ORDER:
    if not any(b == blk for (b, _, _) in cells):
        continue
    print("\n=== %s" % blk)
    for div in DIVS:
        line, vals = "  %-17s" % NICE[div], []
        for s in "012":
            c = cells.get((blk, div, s))
            if c:
                line += "  %5.1f [%4.1f,%4.1f]" % c[:3]; vals.append(c[0])
            else:
                line += "  %16s" % "--"
        print(line + ("   mean %5.1f (%d seeds)" % (sum(vals) / len(vals), len(vals)) if vals else "   --"))

kept = {v[0] for v in used.values()}
print("\n=== judged but not shown (superseded or duplicate judging)")
for f in sorted(set(files) - kept):
    d = json.load(open(f))
    print("  %-58s canonical %.1f" % (f, 100 - d["win_rate"]))
print("\ncells shown: %d" % len(cells))
