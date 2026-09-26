"""Normalization and tokenization of business names and addresses.

String cleanup runs through pandas' Arrow-backed string methods (RE2 regexes, so
no lookarounds and no \\u escapes inside patterns). Tokens are represented as
`Tokens(row, code, vocab)`: one entry per token occurrence, pointing into a
per-chunk vocabulary, so every per-token rule runs once per distinct token.
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

NATIVE = "\u0900-\u0dff"    # Devanagari .. Malayalam
NATIVE_RE = re.compile(f"[{NATIVE}]")

LEGAL = frozenset(
    "pvt private pvtltd ltd limited llp llc lp inc incorporated corp corporation co company "
    "plc opc pllc pc sarl sas sasu sa eurl sci snc gmbh".split())
HONORIFIC = frozenset("mr mrs ms smt shri sri shree the".split())
STOP = frozenset("and of a an de la le les du des et com www net org in at".split())
NAME_DROP = LEGAL | HONORIFIC | STOP
ADDR_DROP = frozenset(
    "street st road rd avenue ave av drive dr lane ln court ct boulevard blvd bd place pl way "
    "highway hwy suite ste unit apt apartment floor flr fl no number plot door house flat shop "
    "building bldg near nr opp opposite po box pmb sector block cross main city township twp "
    "tehsil dist district village rue bis and the of de du la le des null nan none".split())

_ALIAS_PREFIX = (r"^.*?\S[\s(\[]+(?:d\s?/\s?b\s?/\s?a|dba|doing business as|a\s?/\s?k\s?/\s?a|aka"
                 r"|f\s?/\s?k\s?/\s?a|fka|formerly known as)\b[\s:]*")
_NUM_RE = re.compile(r"^\d+(?:[/-][0-9a-z]+)*$")
_LETTER_RE = re.compile(f"[a-z{NATIVE}]")
_UNIT_PREFIX_RE = re.compile(r"^[a-z]{1,2}-(\d)")
_LEAD_ZERO_RE = re.compile(r"^0+(\d)")
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def read_source(path, nrows=None, chunksize=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows, chunksize=chunksize)


def base(s: pd.Series) -> pd.Series:
    return (s.fillna("").str.normalize("NFKD")
            .str.replace("[\u0300-\u036f]", "", regex=True).str.lower())


def clean_name(raw: pd.Series):
    """Return (clean name string, has_alias). For `X dba Y` / `X fka Y` the Y part is kept."""
    s = base(raw)
    stripped = s.str.replace(_ALIAS_PREFIX, "", regex=True)
    has_alias = (stripped != s) & (stripped.str.strip() != "")
    s = stripped.where(has_alias, s)
    s = s.str.replace(r"\bm\s?/\s?s\b", " ", regex=True)
    s = s.str.replace(r"\bl\.?\s?l\.?\s?([pc])\b\.?", r"ll\1", regex=True)
    s = s.str.replace("['\u2019`]", "", regex=True)
    s = s.str.replace("[&+]", " and ", regex=True)
    s = s.str.replace("([a-z])0([a-z])", r"\1o\2", regex=True)
    s = s.str.replace("([a-z])1([a-z])", r"\1l\2", regex=True)
    s = s.str.replace(f"[^0-9a-z{NATIVE}]+", " ", regex=True).str.strip()
    return s, has_alias.to_numpy(dtype=bool)


def clean_addr(raw: pd.Series):
    """Return (clean address string, has_address). `NULL`/`nan` placeholders count as missing."""
    s = base(raw).str.replace(r"\b(?:null|nan|none|n/a)\b", " ", regex=True)
    s = s.str.replace(f"[^0-9a-z{NATIVE}/-]+", " ", regex=True)
    has_addr = s.str.contains(f"[0-9a-z{NATIVE}]", regex=True).to_numpy(dtype=bool)
    return s, has_addr


class Tokens:
    __slots__ = ("row", "code", "vocab")

    def __init__(self, row, code, vocab):
        self.row, self.code, self.vocab = row, code, vocab

    def keep_vocab(self, mask_vocab):
        m = mask_vocab[self.code]
        return Tokens(self.row[m], self.code[m], self.vocab)

    def strings(self):
        return self.vocab[self.code]


def tokenize(s: pd.Series) -> Tokens:
    lists = pc.utf8_split_whitespace(pa.array(s, type=pa.large_string()))
    row = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
    enc = pc.dictionary_encode(pc.list_flatten(lists))
    code = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    vocab = np.asarray(enc.dictionary.to_pylist(), dtype=object)
    return Tokens(row, code, vocab)


def remap(tok: Tokens, fn) -> Tokens:
    """Replace every vocab entry v by the token list fn(v); rows expand or shrink to match."""
    outs = [fn(v) for v in tok.vocab]
    lens = np.fromiter((len(o) for o in outs), np.int64, len(outs))
    starts = np.cumsum(lens) - lens
    rl = lens[tok.code]
    total = int(rl.sum())
    row = np.repeat(tok.row, rl)
    offs = np.arange(total) - np.repeat(np.cumsum(rl) - rl, rl)
    new_code = np.repeat(starts[tok.code], rl) + offs
    labels, uniq = pd.factorize(np.asarray([x for o in outs for x in o], dtype=object))
    return Tokens(row, np.asarray(labels, np.int64)[new_code], np.asarray(uniq, dtype=object))


def _keep(v):
    return [v]


def _addr_norm(v):
    v = v.strip("/-")
    if not v:
        return []
    v = _ORDINAL_RE.sub(r"\1", _UNIT_PREFIX_RE.sub(r"\1", v))
    if v[0] == "0":
        v = _LEAD_ZERO_RE.sub(r"\1", v)
    return [v]


def name_tokens(clean: pd.Series, name_fn=None) -> Tokens:
    t = tokenize(clean)
    if name_fn is not None:
        t = remap(t, name_fn)
    return t.keep_vocab(np.fromiter((v not in NAME_DROP for v in t.vocab), bool, len(t.vocab)))


def addr_tokens(clean: pd.Series, addr_fn=None):
    """Return (word tokens, number tokens) of an address."""
    t = remap(tokenize(clean), _addr_norm)
    if addr_fn is not None:
        t = remap(t, addr_fn)
    isnum = np.fromiter((bool(_NUM_RE.match(v)) for v in t.vocab), bool, len(t.vocab))
    word = np.fromiter(((not n) and len(v) > 1 and v not in ADDR_DROP and bool(_LETTER_RE.search(v))
                        for v, n in zip(t.vocab, isnum)), bool, len(t.vocab))
    return t.keep_vocab(word), t.keep_vocab(isnum)


def has_native(raw: pd.Series) -> np.ndarray:
    return raw.fillna("").str.contains(f"[{NATIVE}]", regex=True).to_numpy(dtype=bool)


# ---- hashing ----------------------------------------------------------------
_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)
_S31, _S29 = np.uint64(31), np.uint64(29)


def str_hash(values) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    if len(values) == 0:
        return np.empty(0, np.uint64)
    return pd.util.hash_array(values, categorize=False)


def mix(a, b):
    """Order-dependent 64-bit combination of two uint64 arrays (or scalars)."""
    with np.errstate(over="ignore"):
        x = (np.asarray(a, np.uint64) ^ (np.asarray(b, np.uint64) * _M1)) * _M2
        x ^= x >> _S31
        x *= _M3
        x ^= x >> _S29
    return x


def country_hash(countries) -> np.ndarray:
    labels, uniq = pd.factorize(np.asarray(countries, dtype=object))
    return str_hash(uniq)[labels]


def token_hash(tok: Tokens, row_country_h: np.ndarray) -> np.ndarray:
    """Hash of (country, token) per occurrence; the same word in two countries never collides."""
    return mix(str_hash(tok.vocab)[tok.code], row_country_h[tok.row])


def dedup_rows(row, h, *extra):
    """Sort by (row, h) and drop repeated tokens within a row."""
    o = np.lexsort((h, row))
    row, h = row[o], h[o]
    keep = np.ones(len(row), bool)
    keep[1:] = (row[1:] != row[:-1]) | (h[1:] != h[:-1])
    return (row[keep], h[keep]) + tuple(e[o][keep] for e in extra)
