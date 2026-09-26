# Business entity resolution — blocking stage

Generates candidate Source 2/3 records for every Source 1 entity. No external data is used;
everything (transliteration dictionary, vocabularies, key frequencies) is learned from the provided files.

## Setup

```
pip install -r requirements.txt
cd src
```

Paths are set in `config.py` and can be overridden with environment variables:
`ER_DATA` (dataset directory), `ER_WORK` (intermediate files), `ER_OUTPUT` (submission output),
`ER_WORKERS` (processes, default 4), `ER_CHUNK` (query rows per task).

## Run

```
python translit.py                 # learn native-script -> Latin dictionary from training pairs (once)
python blocking.py --split train   # candidates for the training data + recall stats inputs
python blocking_report.py --examples
python blocking.py --split test    # writes output/candidate_pairs.tsv
```

`--nrows N` limits every source to its first N rows for quick checks.

## Modules

- `config.py` — paths, caps, key families and weights, train/validation split (`crc32(s1_id) % 10 >= 8` is validation).
- `textnorm.py` — name/address cleaning (legal suffixes, honorifics, dba/aka aliases, `0`→`o` in words),
  Arrow-based tokenization and uint64 token hashing (token hash is country-scoped, so unseen countries such as France work unchanged).
- `translit.py` — Devanagari/Indic tokens mapped to Latin with a dictionary learned from training-split pairs,
  falling back to an `anyascii` consonant skeleton matched against the Source 1 vocabulary.
- `typo.py` — symmetric-delete typo correction and two-word splitting (`faithsfitness` → `faiths fitness`)
  for query tokens missing from the Source 1 vocabulary.
- `blocking.py` — builds blocking keys, indexes Source 1, joins queries in parallel, ranks candidates and applies the final policy.
- `blocking_report.py` — recall, volume and the F0.5 upper bound on the training data.

## Blocking keys

Tokens are ranked by Source 1 document frequency within the country (rarest first). Keys:

| key | content |
|-----|---------|
| NS | one rare name token |
| NP | two name tokens |
| AN | address number + address word |
| AP | two address words |
| NA | name token + address word |
| NN | address number + name token |

Source 1 emits keys from more of its rare tokens than the queries do. Keys shared by more than 50 Source 1
entities (30 for NS) are skipped; a query without an address may use keys up to 200, since it only has name keys.
Candidates are scored by `sum(weight / log2(1 + key frequency)) + W_NAME * Dice(name tokens)`; the overlap
term breaks the frequent ties between e.g. `Sunrise Industries` and `Sunrise Industries Engineering`
(W_NAME = 1 was best on the training data). The kept candidates are the top 5 per query
(top 10 without an address), then at most 200 queries per Source 1 entity.
