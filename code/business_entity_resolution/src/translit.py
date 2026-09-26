"""Transliterate native-script (Indic) tokens into the Latin tokens Source 1 uses.

Native-script names in Sources 2/3 are token-by-token transliterations of the
Source 1 name (`साई आईटी प्राइवेट लिमिटेड` = `Sai IT Private Limited`), and native
tokens inside addresses are state names. Both mappings are learned from the
training ground truth, using only Source 1 entities of the training split so the
validation split stays unseen. Tokens missing from the dictionary fall back to an
anyascii consonant skeleton matched against the Source 1 vocabulary.

Run `python translit.py` to (re)build work/translit.json.
"""
import json
import re
import time
from collections import Counter, defaultdict

import pandas as pd
from anyascii import anyascii

import config
from textnorm import NATIVE_RE, clean_addr, clean_name, has_native, read_source

PATH = config.WORK / "translit.json"

_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
             ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k"), ("x", "ks")]
_COARSE = str.maketrans({"w": "p", "v": "p", "b": "p", "f": "p", "q": "k", "c": "k", "g": "k",
                         "d": "t", "z": "j"})


def skeleton(s: str) -> str:
    """Script-independent consonant skeleton: `सिल्वर` and `silver` both give `slpr`."""
    s = re.sub(r"[^a-z]", "", anyascii(s).lower())
    if not s:
        return ""
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = s.translate(_COARSE)
    s = s[0] + re.sub(r"[aeiouyh]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


class Translit:
    def __init__(self, path=PATH):
        d = json.loads(path.read_text(encoding="utf-8"))
        self.maps = {"name": d["name"], "addr": d["addr"]}
        self.skel = {"name": {}, "addr": {}}

    def set_vocab(self, kind, tokens, counts):
        """Index the Latin S1 vocabulary by skeleton for the fallback (most frequent token wins)."""
        best = {}
        for t, c in zip(tokens, counts):
            if not t.isalpha() or not t.isascii():
                continue
            k = skeleton(t)
            if len(k) >= 3 and c > best.get(k, (0, None))[0]:
                best[k] = (c, t)
        self.skel[kind] = {k: t for k, (_, t) in best.items()}

    def fn(self, kind):
        mapping, skel = self.maps[kind], self.skel[kind]

        def f(v):
            if not NATIVE_RE.search(v):
                return [v]
            out = mapping.get(v)
            if out is not None:
                return out.split()
            t = skel.get(skeleton(v))
            return [t] if t else []
        return f


def _native_pairs():
    rows = []
    for src in (2, 3):
        for ch in read_source(config.query_path("train", src), chunksize=1_000_000):
            rows.append(ch[has_native(ch.business_name) | has_native(ch.business_address)])
    q = pd.concat(rows, ignore_index=True)
    gt = read_source(config.gt_path())
    g = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
    g = g[g.m.isin(q.entity_id)]
    g = g[~g.source1_entity_id.map(config.is_val).astype(bool)]
    s1 = read_source(config.s1_path("train"))
    s1 = s1[s1.entity_id.isin(g.source1_entity_id)].set_index("entity_id")
    return (g.merge(q, left_on="m", right_on="entity_id")
            .join(s1, on="source1_entity_id", rsuffix="_s1"))


def learn():
    t0 = time.time()
    P = _native_pairs()
    print(f"{len(P):,} training-split pairs with native-script text ({time.time() - t0:.0f}s)")

    # names: positional alignment on pairs with equal token counts
    co, cq = Counter(), Counter()
    for a_list, b_list in zip(clean_name(P.business_name)[0].str.split(),
                              clean_name(P.business_name_s1)[0].str.split()):
        if len(a_list) != len(b_list):
            continue
        for a, b in zip(a_list, b_list):
            if NATIVE_RE.search(a) and not NATIVE_RE.search(b):
                co[(a, b)] += 1
                cq[a] += 1
    best = {}
    for (a, b), c in co.items():
        if c >= 2 and c / cq[a] >= 0.5 and c > best.get(a, (0, None))[0]:
            best[a] = (c, b)
    name_map = {a: b for a, (_, b) in best.items()}

    # addresses: Dice co-occurrence between native query tokens and S1 tokens absent from the query
    co, cq, cs = Counter(), Counter(), Counter()
    for a_list, b_list in zip(clean_addr(P.business_address)[0].str.split(),
                              clean_addr(P.business_address_s1)[0].str.split()):
        qs = {a for a in a_list if NATIVE_RE.search(a)}
        if not qs:
            continue
        bs = {b for b in b_list if b.isalpha() and b.isascii()} - set(a_list)
        cq.update(qs)
        cs.update(bs)
        for a in qs:
            for b in bs:
                co[(a, b)] += 1
    cand = defaultdict(list)
    for (a, b), c in co.items():
        dice = 2 * c / (cq[a] + cs[b])
        if c >= 3 and dice >= 0.5:
            cand[a].append((dice, b))
    addr_map = {a: " ".join(b for _, b in sorted(v, reverse=True)[:3]) for a, v in cand.items()}

    config.WORK.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({"name": name_map, "addr": addr_map}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"name map {len(name_map):,} tokens, address map {len(addr_map):,} tokens -> {PATH} ({time.time() - t0:.0f}s)")
    print("address map:", addr_map)


if __name__ == "__main__":
    learn()
