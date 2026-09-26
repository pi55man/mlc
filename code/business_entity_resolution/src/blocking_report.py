"""Blocking quality report for the training split (run `blocking.py --split train` first).

Recall = share of ground-truth (S1, query record) pairs present among the candidates.
`val` columns restrict to the validation S1 entities, which the transliteration
dictionary never saw, so they are the honest numbers.

    python blocking_report.py [--examples]
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from textnorm import read_source


def f05_upper_bound(found, n_true):
    """Macro F0.5 if the matcher were perfect on the candidates (precision 1 wherever something is found)."""
    has = n_true > 0
    r = np.divide(found, n_true, out=np.zeros(len(n_true)), where=has)
    return np.where(has, np.where(found > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true", help="print sample missed pairs with raw text")
    args = ap.parse_args()
    W = config.WORK
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)

    m = pq.read_table(W / "train_qmeta.parquet").to_pandas()
    s1 = pq.read_table(W / "train_s1.parquet").to_pandas()
    s1_val = s1.entity_id.map(config.is_val).to_numpy(dtype=bool)
    n_s1 = len(s1)
    m["matched"] = m.true_s1 >= 0
    m["val"] = m.matched & s1_val[np.maximum(m.true_s1.to_numpy(), 0)]
    m["addr_state"] = np.where(~m.has_addr, "missing", np.where(m.n_addr_tok + m.n_num == 0, "unusable", "ok"))
    m["name_state"] = np.where(m.native_name, "native_script", np.where(m.n_name_tok == 0, "no_known_tokens", "latin"))
    m["k_adapt"] = np.where(m.has_addr, config.K_ADDR, config.K_NO_ADDR)

    fin = pq.read_table(W / "train_cands_final.parquet", columns=["qi", "s1", "is_true"])
    fqi, fs1, ftrue = (fin[c].to_numpy() for c in ("qi", "s1", "is_true"))
    m["found_final"] = False
    m.loc[fqi[ftrue], "found_final"] = True
    m["found_any"] = m.true_rank >= 0

    print("=== 1. Query records (Source 2/3) ===")
    print(m.groupby("src").agg(
        records=("src", "size"), has_addr=("has_addr", "mean"), native_name=("native_name", "mean"),
        alias=("alias", "mean"), typo_corrected=("n_corr", lambda x: (x > 0).mean()),
        zero_keys=("n_keys", lambda x: (x == 0).mean()), keys_mean=("n_keys", "mean"),
        cands_mean=("n_cands", "mean"), matched=("matched", "mean")).round(4).to_string())

    M = m[m.matched]
    V = M[M.val]
    n_true = np.bincount(M.true_s1.to_numpy(), minlength=n_s1)

    c = pq.read_table(W / "train_cands.parquet", columns=["qi", "s1", "rank", "score", "is_true"])
    cqi, cs1, crank, cscore, ctrue = (c[x].to_numpy() for x in ("qi", "s1", "rank", "score", "is_true"))
    del c
    val_pair = s1_val[cs1]

    def row(policy, keep):
        found = np.bincount(cs1[keep & ctrue], minlength=n_s1)
        fb = f05_upper_bound(found, n_true)
        return dict(policy=policy, pairs=int(keep.sum()), recall=found.sum() / len(M),
                    recall_val=(keep & ctrue & val_pair).sum() / len(V), f05_ub=fb.mean(), f05_ub_val=fb[s1_val].mean())

    weak = (~m.has_addr | m.native_name | (m.n_name_tok == 0)).to_numpy()
    top = m.top_score.to_numpy()[cqi]
    rows = [row(f"top-{k}", crank < k) for k in (1, 3, 5)]
    rows.append(row(f"top-{config.K_ADDR} / top-{config.K_NO_ADDR} without address", crank < m.k_adapt.to_numpy()[cqi]))
    rows.append(row("top-5 / top-10 if no address, native name or no known name token", crank < np.where(weak[cqi], 10, 5)))
    rows.append(row("top-5 + ranks 5-9 scoring >= 0.5 x best", (crank < 5) | (cscore >= 0.5 * top)))
    rows.append(row(f"top-{config.K_STORE}", crank < config.K_STORE))
    found_final = np.bincount(fs1[ftrue], minlength=n_s1)
    fb = f05_upper_bound(found_final, n_true)
    rows.append(dict(policy=f"FINAL (per-query top-K + max {config.HUB_CAP} per S1)", pairs=len(fqi),
                     recall=M.found_final.mean(), recall_val=V.found_final.mean(), f05_ub=fb.mean(), f05_ub_val=fb[s1_val].mean()))
    rows.append(dict(policy="all candidates, no cut", pairs=int(m.n_cands.sum()),
                     recall=M.found_any.mean(), recall_val=V.found_any.mean()))
    t = pd.DataFrame(rows)
    t["pairs_per_query"] = t.pairs / len(m)
    t["pair_precision"] = len(M) * t.recall / t.pairs
    tr = [c for c in m.columns if c.startswith("true_rank_w")]
    if tr:
        rows = []
        for c in tr:
            r = M[c]
            ok = r >= 0
            rows.append(dict(w_name=c[len("true_rank_w"):], **{f"top-{k}": (ok & (r < k)).mean() for k in (1, 3, 5, 10)},
                             adaptive=(ok & (r < M.k_adapt)).mean(), adaptive_val=(ok & (r < M.k_adapt))[M.val].mean(),
                             no_addr_top10=(ok & (r < 10))[~M.has_addr].mean()))
        print(f"\n=== 2a. Name-overlap weight (recall of the true pair; current W_NAME={config.W_NAME:g}) ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== 2. Candidate policies (f05_ub = macro F0.5 of a perfect matcher on these candidates) ===")
    print(t.round(4).to_string(index=False))

    print("\n=== 3. Recall by segment (final policy | all candidates) ===")
    for col in ["src", "country", "addr_state", "name_state", "alias"]:
        g = M.groupby(col).agg(n=("found_final", "size"), final=("found_final", "mean"), all=("found_any", "mean"),
                               final_val=("found_final", lambda x: x[M.loc[x.index, "val"]].mean()))
        print(f"\n  by {col}:\n{g.round(4).to_string()}")

    fm = M.true_mask[M.found_any].to_numpy()
    print("\n=== 4. Key families that found the true pair (share of found pairs) ===")
    print(pd.DataFrame({
        "found_by": {n: ((fm >> i) & 1).mean() for i, n in enumerate(config.KEY_TYPES)},
        "found_only_by": {n: (fm == (1 << i)).mean() for i, n in enumerate(config.KEY_TYPES)},
    }).round(4).to_string())

    print("\n=== 5. Source 1 entity level (final policy) ===")
    found = np.bincount(fs1[ftrue], minlength=n_s1)
    cands = np.bincount(fs1, minlength=n_s1)
    fb = f05_upper_bound(found, n_true)
    has = n_true > 0
    print(f"  S1 entities {n_s1:,} ({has.sum():,} with matches, {(~has).sum():,} singletons)")
    print(f"  entities with >=1 true match found: {(found[has] > 0).mean():.4f} (val {(found[has & s1_val] > 0).mean():.4f})")
    print(f"  oracle macro F0.5 upper bound: {fb.mean():.4f} (val {fb[s1_val].mean():.4f})")
    print(f"  candidates per S1: mean {cands.mean():.2f}, p50 {np.median(cands):.0f}, p99 {np.percentile(cands, 99):.0f}, max {cands.max()}")
    print(f"  singletons with 0 candidates: {(cands[~has] == 0).mean():.4f}")

    before = np.bincount(cs1[crank < m.k_adapt.to_numpy()[cqi]], minlength=n_s1)
    hubs = np.argsort(-before)[:8]
    raw = read_source(config.s1_path("train"))
    print(f"\n  largest hubs before the per-S1 cap (S1 entities with > {config.HUB_CAP} candidates: {(before > config.HUB_CAP).sum():,}):")
    print(pd.DataFrame({"candidates": before[hubs], "true_matches": n_true[hubs],
                        "name": raw.business_name.to_numpy()[hubs], "address": raw.business_address.to_numpy()[hubs]}).to_string(index=False))

    if args.examples:
        miss = V[~V.found_final].sample(min(15, (~V.found_final).sum()), random_state=0)
        want = set(miss.entity_id)
        qraw = pd.concat([ch[ch.entity_id.isin(want)] for src in (2, 3)
                          for ch in read_source(config.query_path("train", src), chunksize=1_000_000)]).set_index("entity_id")
        miss = miss.assign(q_name=qraw.business_name.reindex(miss.entity_id).to_numpy(),
                           q_addr=qraw.business_address.reindex(miss.entity_id).to_numpy(),
                           s1_name=raw.business_name.to_numpy()[miss.true_s1], s1_addr=raw.business_address.to_numpy()[miss.true_s1])
        print("\n=== 6. Sample validation pairs missed by the final policy ===")
        print(miss[["q_name", "q_addr", "n_keys", "n_cands", "true_rank", "s1_name", "s1_addr"]].to_string(index=False))


if __name__ == "__main__":
    main()
