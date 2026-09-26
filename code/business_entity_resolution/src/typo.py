"""Map misspelled query tokens onto the Source 1 vocabulary (symmetric-delete lookup).

A query token that never occurs in Source 1 (`hpynmsis`, `trsusell`) cannot form a
blocking key. For each such token we generate its deletion variants and look them
up in an index of the Source 1 vocabulary's deletion variants. A shared variant
means the two words are within (query deletes + vocab deletes) edits, which covers
insertions, deletions, substitutions and adjacent swaps. Among the closest
matches the most frequent vocabulary token wins.
"""
from itertools import combinations

import numpy as np

import config
from textnorm import mix, str_hash


def _deletes(word, max_d):
    out = {word: 0}
    frontier = {word}
    for d in range(1, max_d + 1):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        for w in frontier:
            out.setdefault(w, d)
    return out


def _eligible(tok):
    return len(tok) >= config.TYPO_MIN_LEN and tok.isalpha() and tok.isascii()


def _best_split(toks, todo, country_h, lookup, n_parts, min_part, max_len):
    idx, pieces, bal = [], [], []
    for i in todo:
        tok = toks[i]
        n = len(tok)
        if not (n_parts * min_part <= n <= max_len and tok.isalpha() and tok.isascii()):
            continue
        for cut in combinations(range(min_part, n - min_part + 1), n_parts - 1):
            b = (0, *cut, n)
            lens = [y - x for x, y in zip(b, b[1:])]
            if min(lens) < min_part:
                continue
            idx.append(i)
            pieces.extend(tok[x:y] for x, y in zip(b, b[1:]))
            bal.append(min(lens))
    if not idx:
        return None
    idx = np.asarray(idx, np.int64)
    h = mix(str_hash(pieces), np.repeat(country_h[idx], n_parts)).reshape(-1, n_parts)
    d = lookup(h.ravel()).reshape(-1, n_parts)
    ok = ~np.isnan(d).any(axis=1)
    if not ok.any():
        return None
    idx, h, d, bal = idx[ok], h[ok], d[ok], np.asarray(bal)[ok]
    o = np.lexsort((d.sum(axis=1), -bal, idx))
    idx, h, d = idx[o], h[o], d[o]
    first = np.ones(len(idx), bool)
    first[1:] = idx[1:] != idx[:-1]
    return idx[first], h[first], d[first]


def segment(toks, country_h, lookup, min_part=3):
    """Split run-together words (`faithsfitness`, `quicksevasamiti`) into 2 or 3 vocabulary tokens.

    Returns (token index, part hash, part df) with one entry per part, sorted by token index.
    Among valid splits the most balanced wins, then the rarer parts; 3-way splits are
    only tried for tokens without a 2-way split.
    """
    out_i, out_h, out_d = [], [], []
    todo = np.arange(len(toks))
    for n_parts, max_len in ((2, 60), (3, 24)):
        r = _best_split(toks, todo, country_h, lookup, n_parts, min_part, max_len)
        if r is None:
            continue
        out_i.append(np.repeat(r[0], n_parts))
        out_h.append(r[1].ravel())
        out_d.append(r[2].ravel())
        todo = np.setdiff1d(todo, r[0])
    if not out_i:
        return np.empty(0, np.int64), np.empty(0, np.uint64), np.empty(0)
    i, h, d = np.concatenate(out_i), np.concatenate(out_h), np.concatenate(out_d)
    o = np.argsort(i, kind="stable")
    return i[o], h[o], d[o]


class Corrector:
    def __init__(self, vocab_tok, vocab_country_h, vocab_h, vocab_df, vocab_max_d):
        variants, owner, dist = [], [], []
        for i, tok in enumerate(vocab_tok):
            if not _eligible(tok):
                continue
            for v, d in _deletes(tok, min(vocab_max_d, config.typo_max_dist(len(tok)))).items():
                variants.append(v)
                owner.append(i)
                dist.append(d)
        owner = np.asarray(owner, np.int64)
        vh = mix(str_hash(variants), vocab_country_h[owner])
        o = np.argsort(vh, kind="stable")
        self.vh, self.owner, self.dist = vh[o], owner[o], np.asarray(dist, np.int8)[o]
        self.tok_h = vocab_h
        self.tok_df = vocab_df.astype(np.float64)

    def correct(self, toks, country_h):
        """Return (hash, df) of the correction per token; df is NaN where nothing is close enough."""
        n = len(toks)
        out_h = np.zeros(n, np.uint64)
        out_df = np.full(n, np.nan)
        variants, qi, qd, maxd = [], [], [], []
        for i, tok in enumerate(toks):
            if not _eligible(tok):
                continue
            md = config.typo_max_dist(len(tok))
            for v, d in _deletes(tok, md).items():
                variants.append(v)
                qi.append(i)
                qd.append(d)
                maxd.append(md)
        if not variants:
            return out_h, out_df
        qi = np.asarray(qi, np.int64)
        h = mix(str_hash(variants), country_h[qi])
        lo = np.searchsorted(self.vh, h, "left")
        hi = np.searchsorted(self.vh, h, "right")
        cnt = hi - lo
        if cnt.sum() == 0:
            return out_h, out_df
        rep = np.repeat(np.arange(len(h)), cnt)
        pos = lo[rep] + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        q = qi[rep]
        d = np.asarray(qd, np.int64)[rep] + self.dist[pos]
        ok = (d >= 1) & (d <= np.asarray(maxd, np.int64)[rep])
        q, d, owner = q[ok], d[ok], self.owner[pos[ok]]
        if len(q) == 0:
            return out_h, out_df
        df = self.tok_df[owner]
        o = np.lexsort((owner, -df, d, q))
        q, owner = q[o], owner[o]
        first = np.ones(len(q), bool)
        first[1:] = q[1:] != q[:-1]
        out_h[q[first]] = self.tok_h[owner[first]]
        out_df[q[first]] = self.tok_df[owner[first]]
        return out_h, out_df
