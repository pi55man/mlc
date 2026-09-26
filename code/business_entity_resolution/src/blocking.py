"""Blocking: generate Source 1 candidates for every Source 2/3 ("query") record.

    python blocking.py --split train            # full training split, with ground-truth labels
    python blocking.py --split test             # also writes output/candidate_pairs.tsv
    python blocking.py --split train --nrows 200000   # quick smoke run on the first rows of each file

Pipeline
  1. Source 1 pass A: normalize, tokenize, count token document frequency per country.
  2. Source 1 pass B: emit blocking keys from each record's rarest tokens; build a sorted key index.
  3. Query pass: normalize, transliterate native script, typo-correct or split unknown tokens, emit keys,
     look them up in the index, score candidates by shared rare keys plus name-token overlap,
     keep the top K_STORE.
  4. Final policy: K_ADDR / K_NO_ADDR candidates per query record, at most HUB_CAP per S1 entity.

Outputs in work/: {split}_s1.parquet, {split}_qmeta.parquet, {split}_cands.parquet (top K_STORE),
{split}_cands_final.parquet (after the final policy).
"""
import argparse
import multiprocessing as mp
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from textnorm import (addr_tokens, clean_addr, clean_name, country_hash, dedup_rows, has_native, mix,
                      name_tokens, read_source, str_hash, token_hash)
from translit import Translit
from typo import Corrector, segment

G = {}      # state shared with forked workers
SALT = str_hash(config.KEY_TYPES)
WEIGHTS = np.asarray(config.KEY_WEIGHTS)
T0 = time.time()


def log(*a):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - T0:7.1f}s main-peak {peak:4.2f}GB]", *a, flush=True)


# ---- per-record token preparation ---------------------------------------------

def _prepare(df, translit):
    ch = country_hash(df.country.to_numpy(dtype=object))
    nclean, alias = clean_name(df.business_name)
    aclean, has_addr = clean_addr(df.business_address)
    nt = name_tokens(nclean, translit.fn("name") if translit else None)
    at, nums = addr_tokens(aclean, translit.fn("addr") if translit else None)
    return ch, alias, has_addr, nt, at, nums


class VocabIndex:
    def __init__(self, h, df):
        self.h, self.df = h, df.astype(np.float64)

    def lookup(self, q):
        if len(self.h) == 0:
            return np.full(len(q), np.nan)
        pos = np.minimum(np.searchsorted(self.h, q), len(self.h) - 1)
        return np.where(self.h[pos] == q, self.df[pos], np.nan)


def _resolve(t, ch, kind, correct):
    """Token occurrences -> (row, hash, S1 document frequency, corrected flag), in-vocabulary only.

    With `correct`, tokens unknown to Source 1 are typo-corrected, or else split into two known words.
    """
    row, h, code = dedup_rows(t.row, token_hash(t, ch), t.code)
    vi = G[f"vi_{kind}"]
    df = vi.lookup(h)
    corrected = np.zeros(len(h), bool)
    if correct:
        oov = np.flatnonzero(np.isnan(df))
        if len(oov):
            _, first, inv = np.unique(h[oov], return_index=True, return_inverse=True)
            toks, tch = t.vocab[code[oov[first]]], ch[row[oov[first]]]
            nh, ndf = G[f"corr_{kind}"].correct(toks, tch)
            still = np.flatnonzero(np.isnan(ndf))
            si, sh, sd = segment(toks[still], tch[still], vi.lookup)
            fixed = ~np.isnan(ndf[inv])
            h[oov] = np.where(fixed, nh[inv], h[oov])
            df[oov] = ndf[inv]
            corrected[oov] = fixed
            if len(si):
                n_part = np.bincount(still[si], minlength=len(first))
                start = np.cumsum(n_part) - n_part
                c = n_part[inv]
                occ = np.repeat(np.arange(len(oov)), c)
                pos = start[inv][occ] + np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
                row = np.concatenate([row, row[oov[occ]]])
                h = np.concatenate([h, sh[pos]])
                df = np.concatenate([df, sd[pos]])
                corrected = np.concatenate([corrected, np.ones(len(pos), bool)])
            row, h, df, corrected = dedup_rows(row, h, df, corrected)
    m = ~np.isnan(df)
    return row[m], h[m], df[m], corrected[m]


def _rank(row):
    """Position of each element inside its run of equal `row` values (row must be sorted)."""
    n = len(row)
    if n == 0:
        return np.empty(0, np.int64)
    start = np.ones(n, bool)
    start[1:] = row[1:] != row[:-1]
    idx = np.arange(n)
    return idx - np.maximum.accumulate(np.where(start, idx, 0))


def _size(row):
    n = len(row)
    if n == 0:
        return np.empty(0, np.int64)
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    counts = np.diff(np.r_[starts, n])
    return np.repeat(counts, counts)


def _nums(t, ch):
    row, h = dedup_rows(t.row, token_hash(t, ch))
    m = _rank(row) < config.MAX_NUMS
    return row[m], h[m]


# ---- blocking keys -----------------------------------------------------------------

def _pair_keys(row, h, rank, size, n, typ):
    rs, ks = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sel = np.flatnonzero((rank == i) & (size > j))
            a, b = h[sel], h[sel + (j - i)]
            rs.append(row[sel])
            ks.append(mix(mix(np.minimum(a, b), np.maximum(a, b)), SALT[typ]))
    if not rs:
        return np.empty(0, np.int64), np.empty(0, np.uint64)
    return np.concatenate(rs), np.concatenate(ks)


def _cross_keys(ra, ha, rb, hb, typ):
    if len(ra) == 0 or len(rb) == 0:
        return np.empty(0, np.int64), np.empty(0, np.uint64)
    m = pd.DataFrame({"r": ra, "a": ha}).merge(pd.DataFrame({"r": rb, "b": hb}), on="r")
    return m.r.to_numpy(np.int64), mix(mix(m.a.to_numpy(np.uint64), m.b.to_numpy(np.uint64)), SALT[typ])


def build_keys(name, addr, nums, side):
    """Emit (row, key, key type) for the five key families, all scoped to the record's country."""
    p = config.TOP[side]
    o = np.lexsort((name[1], name[2], name[0]))
    nr, nh, ndf = name[0][o], name[1][o], name[2][o]
    o = np.lexsort((addr[1], addr[2], addr[0]))
    ar, ah = addr[0][o], addr[1][o]
    nrank, nsize = _rank(nr), _size(nr)
    arank, asize = _rank(ar), _size(ar)
    parts = []
    m = ndf <= config.CAP_NO_ADDR
    parts.append((nr[m], mix(nh[m], SALT[0]), 0))
    parts.append((*_pair_keys(nr, nh, nrank, nsize, p["np_"], 1), 1))
    m = arank < p["an"]
    parts.append((*_cross_keys(nums[0], nums[1], ar[m], ah[m], 2), 2))
    parts.append((*_pair_keys(ar, ah, arank, asize, p["ap"], 3), 3))
    mn, ma = nrank < p["na_n"], arank < p["na_a"]
    parts.append((*_cross_keys(nr[mn], nh[mn], ar[ma], ah[ma], 4), 4))
    mn = nrank < p["nn"]
    parts.append((*_cross_keys(nums[0], nums[1], nr[mn], nh[mn], 5), 5))
    rid = np.concatenate([x[0] for x in parts]).astype(np.int64)
    key = np.concatenate([x[1] for x in parts]).astype(np.uint64)
    typ = np.concatenate([np.full(len(x[0]), x[2], np.int8) for x in parts])
    return rid, key, typ


# ---- Source 1 ----------------------------------------------------------------------

def _s1_vocab_task(rng):
    a, b = rng
    ch, _, _, nt, at, _ = _prepare(G["s1"].iloc[a:b], None)
    out = {}
    for kind, t in (("name", nt), ("addr", at)):
        row, h, code = dedup_rows(t.row, token_hash(t, ch), t.code)
        uh, first, cnt = np.unique(h, return_index=True, return_counts=True)
        out[kind] = (uh, cnt, t.vocab[code[first]], ch[row[first]])
    return out


def _merge_vocab(parts, kind):
    h = np.concatenate([p[kind][0] for p in parts])
    cnt = np.concatenate([p[kind][1] for p in parts])
    tok = np.concatenate([p[kind][2] for p in parts])
    ch = np.concatenate([p[kind][3] for p in parts])
    uh, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return uh, np.bincount(inv, weights=cnt).astype(np.int64), tok[first], ch[first]


def _s1_keys_task(rng):
    a, b = rng
    ch, _, _, nt, at, nums = _prepare(G["s1"].iloc[a:b], None)
    name = _resolve(nt, ch, "name", False)[:3]
    addr = _resolve(at, ch, "addr", False)[:3]
    rid, key, typ = build_keys(name, addr, _nums(nums, ch), "s1")
    return rid + a, key, typ, np.bincount(rid, minlength=b - a), name[0] + a, name[1]


def _ranges(n):
    return [(a, min(a + config.CHUNK, n)) for a in range(0, n, config.CHUNK)]


def build_s1(split, nrows, workers, ctx):
    s1 = read_source(config.s1_path(split), nrows=nrows)
    G["s1"] = s1
    log(f"S1 loaded: {len(s1):,} records")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_s1_vocab_task, _ranges(len(s1)))
    for kind in ("name", "addr"):
        h, df, tok, ch = _merge_vocab(parts, kind)
        G[f"vi_{kind}"] = VocabIndex(h, df)
        G[f"vocab_{kind}"] = (h, df, tok, ch)
    del parts
    log(f"vocab: name {len(G['vi_name'].h):,} tokens, address {len(G['vi_addr'].h):,} tokens")

    with ctx.Pool(workers) as pool:
        parts = pool.map(_s1_keys_task, _ranges(len(s1)))
    rid = np.concatenate([p[0] for p in parts])
    key = np.concatenate([p[1] for p in parts])
    typ = np.concatenate([p[2] for p in parts])
    s1_nkeys = np.concatenate([p[3] for p in parts])
    name_row = np.concatenate([p[4] for p in parts])
    G["S1N"] = np.sort(mix(name_row.astype(np.uint64), np.concatenate([p[5] for p in parts])))
    G["S1_NN"] = np.bincount(name_row, minlength=len(s1)).astype(np.int32)
    del parts, name_row
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    freq = cnt[inv]
    del inv, cnt
    print(f"  S1 keys per record: mean {s1_nkeys.mean():.1f}, zero-key records {(s1_nkeys == 0).mean():.4%}")
    for t, name in enumerate(config.KEY_TYPES):
        sel = typ == t
        print(f"  {name}: {sel.sum():>11,} key rows, {(freq[sel] > config.CAP_NO_ADDR).mean():.4f} dropped as too generic")
    keep = freq <= config.CAP_NO_ADDR
    o = np.argsort(key[keep], kind="stable")
    G["K1"] = key[keep][o]
    G["S1R"] = rid[keep][o].astype(np.int32)
    G["TY1"] = typ[keep][o]
    G["F1"] = freq[keep][o].astype(np.int32)
    G["n_s1"] = len(s1)
    log(f"S1 key index: {len(G['K1']):,} entries")

    s1_meta = pd.DataFrame({"entity_id": s1.entity_id, "country": s1.country})
    del G["s1"]
    return s1_meta


def build_query_helpers(s1_meta, split):
    tr = Translit()
    for kind in ("name", "addr"):
        h, df, tok, ch = G[f"vocab_{kind}"]
        tr.set_vocab(kind, tok, df)
        G[f"corr_{kind}"] = Corrector(tok, ch, h, df, vocab_max_d=2 if kind == "name" else 1)
        del G[f"vocab_{kind}"]
    G["translit"] = tr
    G["gmap"] = None
    if split == "train":
        gt = read_source(config.gt_path())
        g = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
        g = g[g.m.notna() & (g.m != "")]
        s1_idx = pd.Index(s1_meta.entity_id).get_indexer(g.source1_entity_id)
        ok = s1_idx >= 0
        code = _qcode(g.m)[ok]
        o = np.argsort(code)
        G["gmap"] = (code[o], s1_idx[ok][o].astype(np.int64))
    log("query helpers ready (transliteration, typo correction" + (", ground truth)" if G["gmap"] is not None else ")"))


def _qcode(ids: pd.Series) -> np.ndarray:
    return ids.str.slice(1, 2).astype(np.int64).to_numpy() * 10**10 + ids.str.slice(3).astype(np.int64).to_numpy()


# ---- queries -----------------------------------------------------------------------

def _shared_names(pq_, ps1, nr, nh, n):
    """Number of name tokens each (query, S1) candidate pair has in common."""
    o = np.argsort(nr, kind="stable")
    nr, nh = nr[o], nh[o]
    cnt = np.bincount(nr, minlength=n)
    start = np.cumsum(cnt) - cnt
    c = cnt[pq_]
    pair = np.repeat(np.arange(len(pq_)), c)
    pos = start[pq_][pair] + np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
    key = mix(ps1[pair].astype(np.uint64), nh[pos])
    S = G["S1N"]
    hit = S[np.minimum(np.searchsorted(S, key), len(S) - 1)] == key
    return np.bincount(pair[hit], minlength=len(pq_))


def _query_task(rng):
    a, b = rng
    df = G["q"].iloc[a:b]
    n = b - a
    ch, alias, has_addr, nt, at, nums = _prepare(df, G["translit"])
    nr, nh, ndf, ncorr = _resolve(nt, ch, "name", True)
    ar, ah, adf, acorr = _resolve(at, ch, "addr", True)
    numr, numh = _nums(nums, ch)
    qr, qk, _ = build_keys((nr, nh, ndf), (ar, ah, adf), (numr, numh), "q")

    K1, n_s1 = G["K1"], G["n_s1"]
    lo, hi = np.searchsorted(K1, qk, "left"), np.searchsorted(K1, qk, "right")
    cnt = hi - lo
    qi = np.repeat(np.arange(len(qk)), cnt)
    pos = lo[qi] + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    q, s1, t, f = qr[qi], G["S1R"][pos].astype(np.int64), G["TY1"][pos].astype(np.int64), G["F1"][pos]
    ha = has_addr[q]
    cap = np.where(ha, np.where(t == 0, config.CAP_NAME_SINGLE, config.CAP), config.CAP_NO_ADDR)
    ok = f <= cap
    q, s1, t, f = q[ok], s1[ok], t[ok], f[ok]

    pk = q * n_s1 + s1
    upk, inv = np.unique(pk, return_inverse=True)
    key_score = np.bincount(inv, weights=WEIGHTS[t] / np.log2(1.0 + f), minlength=len(upk))
    mask = np.zeros(len(upk), np.int8)
    np.bitwise_or.at(mask, inv, (1 << t).astype(np.int8))
    pq_, ps1 = upk // n_s1, upk % n_s1
    nq = np.bincount(nr, minlength=n)
    dice = 2.0 * _shared_names(pq_, ps1, nr, nh, n) / np.maximum(nq[pq_] + G["S1_NN"][ps1], 1)
    score = key_score + config.W_NAME * dice
    o = np.lexsort((ps1, -score, pq_))
    pq_, ps1, score, mask, key_score, dice = (x[o] for x in (pq_, ps1, score, mask, key_score, dice))
    ncand = np.bincount(pq_, minlength=n)
    rank = np.arange(len(pq_)) - (np.cumsum(ncand) - ncand)[pq_]
    top = np.zeros(n)
    top[pq_[rank == 0]] = score[rank == 0]

    meta = dict(
        entity_id=df.entity_id.to_numpy(dtype=object), country=df.country.to_numpy(dtype=object),
        has_addr=has_addr, native_name=has_native(df.business_name), alias=alias,
        n_name_tok=np.bincount(nr, minlength=n).astype(np.int16),
        n_addr_tok=np.bincount(ar, minlength=n).astype(np.int16),
        n_num=np.bincount(numr, minlength=n).astype(np.int8),
        n_corr=(np.bincount(nr[ncorr], minlength=n) + np.bincount(ar[acorr], minlength=n)).astype(np.int8),
        n_keys=np.bincount(qr, minlength=n).astype(np.int16),
        n_cands=ncand.astype(np.int32), top_score=top.astype(np.float32))
    keep = rank < config.K_STORE
    cands = dict(qi=pq_[keep].astype(np.int32), s1=ps1[keep].astype(np.int32), rank=rank[keep].astype(np.int8),
                 score=score[keep].astype(np.float32), mask=mask[keep])
    if G["gmap"] is not None:
        gcode, gs1 = G["gmap"]
        qc = _qcode(df.entity_id)
        j = np.minimum(np.searchsorted(gcode, qc), len(gcode) - 1)
        tq = np.where(gcode[j] == qc, gs1[j], -1)
        is_true = ps1 == tq[pq_]
        true_rank = np.full(n, -1, np.int32)
        true_rank[pq_[is_true]] = rank[is_true]
        true_mask = np.zeros(n, np.int8)
        true_mask[pq_[is_true]] = mask[is_true]
        meta.update(true_s1=tq.astype(np.int32), true_rank=true_rank, true_mask=true_mask)
        for w in config.W_NAME_TRY:
            s = key_score + w * dice
            ts = np.full(n, np.inf)
            ts[pq_[is_true]] = s[is_true]
            ahead = (s > ts[pq_]) | ((s == ts[pq_]) & (ps1 < tq[pq_]))
            r = np.bincount(pq_, weights=ahead, minlength=n).astype(np.int32)
            meta[f"true_rank_w{w:g}"] = np.where(true_rank >= 0, r, -1)
        cands["is_true"] = is_true[keep]
    return meta, cands


def run_queries(split, nrows, workers, ctx):
    meta_w = cand_w = None
    offset = 0
    for src in (2, 3):
        G["q"] = read_source(config.query_path(split, src), nrows=nrows)
        log(f"S{src} loaded: {len(G['q']):,} records")
        with ctx.Pool(workers) as pool:
            for i, (meta, cands) in enumerate(pool.imap(_query_task, _ranges(len(G["q"])))):
                meta["src"] = np.full(len(meta["entity_id"]), src, np.int8)
                cands["qi"] = cands["qi"] + offset
                offset += len(meta["entity_id"])
                mt, ct = pa.table(meta), pa.table(cands)
                if meta_w is None:
                    meta_w = pq.ParquetWriter(config.WORK / f"{split}_qmeta.parquet", mt.schema)
                    cand_w = pq.ParquetWriter(config.WORK / f"{split}_cands.parquet", ct.schema)
                meta_w.write_table(mt)
                cand_w.write_table(ct)
                if i % 10 == 9:
                    log(f"  S{src}: {offset:,} query records done")
        del G["q"]
    meta_w.close()
    cand_w.close()
    log(f"queries done: {offset:,} records")


# ---- final candidate policy ------------------------------------------------------------

def final_policy(split):
    """Top K_ADDR (K_NO_ADDR without an address) per query record, then at most HUB_CAP per S1 entity."""
    has_addr = pq.read_table(config.WORK / f"{split}_qmeta.parquet", columns=["has_addr"])["has_addr"].to_numpy()
    t = pq.read_table(config.WORK / f"{split}_cands.parquet")
    k = np.where(has_addr[t["qi"].to_numpy()], config.K_ADDR, config.K_NO_ADDR)
    t = t.filter(pa.array(t["rank"].to_numpy() < k))
    s1 = t["s1"].to_numpy()
    o = np.lexsort((-t["score"].to_numpy(), s1))
    keep = np.sort(o[_rank(s1[o]) < config.HUB_CAP])
    t = t.take(pa.array(keep))
    pq.write_table(t, config.WORK / f"{split}_cands_final.parquet")
    log(f"final candidates: {t.num_rows:,} pairs")
    return t


def write_candidate_tsv(split, final, path):
    s1_ids = pq.read_table(config.WORK / f"{split}_s1.parquet", columns=["entity_id"])["entity_id"].to_numpy(zero_copy_only=False)
    q_ids = pq.read_table(config.WORK / f"{split}_qmeta.parquet", columns=["entity_id"])["entity_id"].to_numpy(zero_copy_only=False)
    s1 = final["s1"].to_numpy()
    o = np.argsort(s1, kind="stable")
    s1, qi = s1[o], final["qi"].to_numpy()[o]
    bounds = np.searchsorted(s1, np.arange(len(s1_ids) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        step = 100_000
        for a in range(0, len(s1_ids), step):
            b = min(a + step, len(s1_ids))
            lo, hi = bounds[a], bounds[b]
            joined = pd.Series(q_ids[qi[lo:hi]]).groupby(s1[lo:hi] - a).agg(",".join)
            col = np.full(b - a, "", dtype=object)
            col[joined.index.to_numpy()] = joined.to_numpy()
            f.write("".join(f"{e}\t{c}\n" for e, c in zip(s1_ids[a:b], col)))
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--nrows", type=int, default=None, help="only read the first N rows of each source (smoke runs)")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()
    config.WORK.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")

    s1_meta = build_s1(args.split, args.nrows, args.workers, ctx)
    s1_meta.to_parquet(config.WORK / f"{args.split}_s1.parquet", index=False)
    build_query_helpers(s1_meta, args.split)
    run_queries(args.split, args.nrows, args.workers, ctx)
    final = final_policy(args.split)
    if args.split == "test":
        write_candidate_tsv(args.split, final, config.OUTPUT / "candidate_pairs.tsv")
    log("done")


if __name__ == "__main__":
    main()
