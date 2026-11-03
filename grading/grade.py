"""Grading batches and the results table.

  uv run grading/grade.py make NAME [...]     replies/NAME.jsonl -> grades/NAME/sec_<section>.jsonl (tier-1 sections casual,
                                              task, identity, world; the battery as one section); prints the args for the
                                              Claude Code workflow grading/workflow.js, which writes grades/NAME/labels_<section>.json
  uv run grading/grade.py table NAME [...]    per model and prompt set (or tier-1 section with --sections): Haiku's grades
                                              (fidelity, assistant register, hard breaks, factual consistency), runaway replies
                                              (> 2,000 characters), and formula measures (replies sharing a 3-word opener, a
                                              closing sentence; replies ending on a question)
  --root reference   reads reference/replies + reference/grades (our runs) instead of replies/ + grades/"""
import argparse, collections, json, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("make", "table")); ap.add_argument("names", nargs="+")
ap.add_argument("--root", default="."); ap.add_argument("--sections", action="store_true", help="table: tier-1 sections separately")
a = ap.parse_args(); R = Path(a.root)
SECS = ("casual", "task", "identity", "world", "battery")
read = lambda f: [json.loads(l) for l in Path(f).read_text().split("\n") if l.strip()]

if a.cmd == "make":
    for n in a.names:
        rows = read(R / "replies" / f"{n}.jsonl"); d = R / "grades" / n; d.mkdir(parents=True, exist_ok=True); made = []
        for s in SECS:
            part = [r for r in rows if (r["set"] == "battery") == (s == "battery") and (s == "battery" or r["section"] == s)]
            if part:
                (d / f"sec_{s}.jsonl").write_text("".join(json.dumps({"id": r["id"], "q": r["prompt"], "reply": r["reply"]}, ensure_ascii=False) + "\n" for r in part))
                made.append(s)
        print(f"{d}:{'+'.join(made)}")
    raise SystemExit

words = lambda t: re.findall(r"[a-z']+", t.lower())
def shared(keys):                                    # (share of replies whose key matches another reply's)
    c = collections.Counter(keys); return sum(v for v in c.values() if v > 1) / max(1, len(keys))
mean = lambda v: sum(v) / len(v) if v else float("nan")
groups = (("tier1", ("casual", "task", "identity", "world")), ("battery", ("battery",)))
if a.sections: groups = tuple((s, (s,)) for s in SECS)
print(f"{'model':16s} {'set':9s} {'n':>4s} | {'fidelity':>8s} {'assist':>6s} {'break':>6s} {'factual':>7s} | {'runaway':>7s} | "
      f"{'shared opener':>13s} {'shared closer':>13s} {'ends ?':>6s}")
for n in a.names:
    rows = read(R / "replies" / f"{n}.jsonl"); P = collections.defaultdict(dict)   # (pass -> {(section, id): grade})
    for s in SECS:                                   # (grading passes: labels_<s>.json and labels_<s>.<pass>.json)
        for f in sorted((R / "grades" / n).glob(f"labels_{s}*.json")):
            P[f.name[len(f"labels_{s}"):-5].lstrip(".") or "main"].update({(s, int(k)): v for k, v in json.loads(f.read_text()).items()})
    key = lambda r: (r["section"] if r["set"] == "tier1" else "battery", r["id"])
    for gname, secs in groups:
        part = [r for r in rows if key(r)[0] in secs]
        if not part: continue
        def m(f, gs): return mean([f(x) for x in gs if f(x) is not None])
        per = [[L[key(r)] for r in part if key(r) in L] for L in P.values()]       # (each pass's grades for this group)
        def cell(f, fmt):                            # (mean over passes, and the spread between passes when there are two or more)
            v = [m(f, g) for g in per if g]
            if not v: return "—"
            return fmt(mean(v)) + (f"±{(max(v) - min(v)) / 2:.2f}" if len(v) > 1 else "")
        reps = [r["reply"].strip() for r in part]
        last = [" ".join(words((re.split(r"(?<=[.?!])\s+", x) or [""])[-1])) for x in reps]
        print(f"{n:16s} {gname:9s} {len(part):4d} | {cell(lambda x: x['fidelity'], '{:.2f}'.format):>8s} "
              f"{cell(lambda x: x['assistant'], '{:.2f}'.format):>6s} {cell(lambda x: float(x['hard_break']), '{:.0%}'.format):>6s} "
              f"{cell(lambda x: x['factual'], '{:.2f}'.format):>7s} | {sum(len(x) > 2000 for x in reps):7d} | "
              f"{shared([' '.join(words(x)[:3]) for x in reps]):13.0%} {shared(last):13.0%} {mean([float(x.endswith('?')) for x in reps]):6.0%}"
              + (f"   ({len(P)} grading passes)" if len(P) > 1 else ""))
