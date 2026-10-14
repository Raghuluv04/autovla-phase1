"""BLEU-4 / METEOR / CIDEr-D for explanation quality (Milestone 5).

Dependency-free so the benchmark runs anywhere. Deviations from the reference
implementations are called out per metric; numbers are therefore comparable
*between our runs* and approximately (not exactly) with published ones.

  BLEU-4   corpus-level, closest-ref brevity penalty, optional add-one smoothing
  METEOR   original Banerjee-Lavie formulation (exact + stem matching, Fmean with
           recall weight 9, fragmentation penalty 0.5*(chunks/matches)^3).
           No WordNet synonym stage (needs a download), so scores run slightly
           lower than METEOR 1.5.
  CIDEr-D  follows pycocoevalcap: tf-idf n-grams (n=1..4), Gaussian length
           penalty (sigma=6), clipped numerator, x10; idf from the reference set.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
CIDER_SIGMA = 6.0
METEOR_ALPHA_RECALL = 9.0
METEOR_GAMMA = 0.5
METEOR_BETA = 3.0
MAX_N = 4


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------
def bleu(hyps: list[str], refs: list[list[str]], max_n: int = MAX_N, smooth: bool = False) -> float:
    """Corpus BLEU. ``refs[i]`` is the list of references for ``hyps[i]``."""
    match, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for h, rs in zip(hyps, refs):
        ht = tokenize(h)
        rts = [tokenize(r) for r in rs]
        hyp_len += len(ht)
        ref_len += min((abs(len(r) - len(ht)), len(r)) for r in rts)[1]
        for n in range(1, max_n + 1):
            hc = _ngrams(ht, n)
            mx = Counter()
            for r in rts:
                for g, c in _ngrams(r, n).items():
                    mx[g] = max(mx[g], c)
            match[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            total[n - 1] += max(len(ht) - n + 1, 0)
    if hyp_len == 0 or any(t == 0 for t in total):
        return 0.0
    logs = []
    for n, (m, t) in enumerate(zip(match, total), start=1):
        k = 1 if (smooth and n > 1) else 0       # add-one on n>1 only (sacrebleu 'add-k')
        if m + k == 0:
            return 0.0
        logs.append(math.log((m + k) / (t + k)))
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(sum(logs) / max_n)


# ---------------------------------------------------------------------------
# METEOR (exact + stem)
# ---------------------------------------------------------------------------
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s")


def stem(w: str) -> str:
    """Light suffix stripper (stand-in for Porter; deterministic and offline)."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def _align(h, r):
    """Greedy two-stage (exact, then stem) alignment -> list of (hyp_i, ref_j)."""
    used_r, pairs = set(), []
    for key in (lambda w: w, stem):
        used_h = {i for i, _ in pairs}
        for i, w in enumerate(h):
            if i in used_h:
                continue
            for j, v in enumerate(r):
                if j not in used_r and key(w) == key(v):
                    pairs.append((i, j)); used_r.add(j)
                    break
    return sorted(pairs)


def _meteor_single(h, r) -> float:
    if not h or not r:
        return 0.0
    pairs = _align(h, r)
    m = len(pairs)
    if m == 0:
        return 0.0
    p, rec = m / len(h), m / len(r)
    fmean = (1 + METEOR_ALPHA_RECALL) * p * rec / (rec + METEOR_ALPHA_RECALL * p)
    chunks = 1 + sum(1 for (i0, j0), (i1, j1) in zip(pairs, pairs[1:])
                     if not (i1 == i0 + 1 and j1 == j0 + 1))
    penalty = METEOR_GAMMA * (chunks / m) ** METEOR_BETA
    return fmean * (1 - penalty)


def meteor(hyps: list[str], refs: list[list[str]]) -> float:
    """Mean over segments of the best score across that segment's references."""
    scores = [max(_meteor_single(tokenize(h), tokenize(r)) for r in rs)
              for h, rs in zip(hyps, refs)]
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# CIDEr-D
# ---------------------------------------------------------------------------
def cider(hyps: list[str], refs: list[list[str]], n: int = MAX_N, sigma: float = CIDER_SIGMA) -> float:
    ref_toks = [[tokenize(r) for r in rs] for rs in refs]
    df = [Counter() for _ in range(n)]
    for rs in ref_toks:
        for k in range(n):
            seen = set()
            for r in rs:
                seen.update(_ngrams(r, k + 1))
            df[k].update(seen)
    log_n = math.log(float(len(refs)))

    def vec(tokens):
        vs, norms = [], []
        for k in range(n):
            counts = _ngrams(tokens, k + 1)
            v = {g: c * (log_n - math.log(max(1.0, df[k].get(g, 0.0)))) for g, c in counts.items()}
            vs.append(v)
            norms.append(math.sqrt(sum(x * x for x in v.values())))
        return vs, norms

    scores = []
    for h, rs in zip(hyps, ref_toks):
        ht = tokenize(h)
        hv, hn = vec(ht)
        acc = [0.0] * n
        for r in rs:
            rv, rn = vec(r)
            delta = len(ht) - len(r)
            for k in range(n):
                val = sum(min(x, rv[k].get(g, 0.0)) * rv[k].get(g, 0.0) for g, x in hv[k].items())
                if hn[k] and rn[k]:
                    val /= hn[k] * rn[k]
                else:
                    val = 0.0
                acc[k] += val * math.exp(-(delta ** 2) / (2 * sigma ** 2))
        scores.append(sum(a / len(rs) for a in acc) / n * 10.0)
    return sum(scores) / max(len(scores), 1)


@dataclass
class ExplanationScores:
    bleu4: float
    meteor: float
    cider: float
    n: int

    def as_dict(self):
        return dict(BLEU4=self.bleu4, METEOR=self.meteor, CIDEr=self.cider, n=self.n)


def score_explanations(hyps: list[str], refs: list[list[str]]) -> ExplanationScores:
    if len(hyps) != len(refs) or not hyps:
        raise ValueError("need equally many (>0) hypotheses and reference lists")
    return ExplanationScores(bleu(hyps, refs, smooth=True), meteor(hyps, refs), cider(hyps, refs), len(hyps))


# ---------------------------------------------------------------------------
# BDD-X references
# ---------------------------------------------------------------------------
def load_bddx_references(path) -> dict[str, list[str]]:
    """Read BDD-X-Annotations_v1.csv -> {clip_segment_id: [justification, ...]}.

    The Mechanical-Turk export has one row per video with repeated column groups
    ``Answer.<k>action`` / ``Answer.<k>justification``; this also accepts a flat
    ``id,action,justification`` CSV. Each (video, k) pair becomes one item whose
    references are the action and its justification sentence.
    """
    out: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        for row_i, row in enumerate(csv.DictReader(f)):
            if "justification" in row:
                out[row.get("id") or str(row_i)] = [t for t in (row.get("justification"), ) if t]
                continue
            vid = row.get("Input.Video") or str(row_i)
            for key, val in row.items():
                m = re.fullmatch(r"Answer\.(\d+)justification", key)
                if m and val and val.strip():
                    out[f"{vid}#{m.group(1)}"] = [val.strip()]
    return out


def main():
    """Score model explanations against BDD-X.

    ``--hyps`` is a JSONL of ``{"id": "<video>#<k>", "text": "..."}`` whose ids
    match :func:`load_bddx_references` keys (fetch the CSV with
    ``data_pipeline/scripts/download_bddx.sh``).
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--bddx", required=True)
    ap.add_argument("--hyps", required=True)
    args = ap.parse_args()
    refs = load_bddx_references(args.bddx)
    hyps = {}
    with open(args.hyps) as f:
        for line in f:
            d = json.loads(line)
            hyps[d["id"]] = d["text"]
    ids = [i for i in hyps if i in refs]
    if not ids:
        raise SystemExit("no hypothesis id matches a BDD-X reference id")
    print(score_explanations([hyps[i] for i in ids], [refs[i] for i in ids]).as_dict())


if __name__ == "__main__":
    main()
