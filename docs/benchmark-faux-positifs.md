# False positives on real models (step 9)

Nitrogen scanned against a frozen list of popular, trusted Hugging Face
models, each pinned to an exact commit revision ([`scripts/models.txt`](../scripts/models.txt)).
Every model is benign, so any RISK verdict is a false positive. The
benchmark needs the network, so it is opt-in and never part of the test
suite:

```
uv run python ../scripts/benchmark_false_positives.py \
    --models ../scripts/models.txt --out ../docs/benchmark-faux-positifs.md
```

Snapshot of 2026-10-23: **18 models** evaluated, out of a 100-model pinned list (the rest exceed the default 1.6 GiB size
budget per model and are reproducible by raising `--max-bytes`). A model is
never loaded or deserialized; only the scanner's verdict is read.

## Result

- Models evaluated: **18**
- **False positives: 0 / 18 (0%)**

Every evaluated model is COMPLIANT.

## Impact of ADR 0009 and ADR 0010

Before the two fixes this benchmark drove, **11 of the same 18 models (61%)** were
false-positived — every one of them on raw tensor **storage** (a
`data/<key>` member, or the storage a legacy `.bin` appends after its
pickles), never on the real pickle (`data.pkl`). Detecting storage bytes as
a tar ([ADR 0009](adr/0009-detection-archive-imbriquee.md)) or as a pickle /
oversized / nested archive ([ADR 0010](adr/0010-membres-de-stockage-pytorch.md))
was the cause; the fixes analyze only what a loader actually unpickles.

## Per-model detail

| Model | Revision | Before | After |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405e` | ❌ RISK (pickle.unanalyzable) | ✅ COMPLIANT |
| `Qwen/Qwen3-0.6B` | `c1899de289a0` | COMPLIANT | ✅ COMPLIANT |
| `amazon/chronos-2` | `29ec3766d36d` | COMPLIANT | ✅ COMPLIANT |
| `autogluon/chronos-2` | `60088152a34e` | COMPLIANT | ✅ COMPLIANT |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c44` | COMPLIANT | ✅ COMPLIANT |
| `google-bert/bert-base-uncased` | `86b5e0934494` | ❌ RISK (pickle.malformed) | ✅ COMPLIANT |
| `google-t5/t5-small` | `df1b051c4962` | ❌ RISK (pickle.malformed) | ✅ COMPLIANT |
| `google/electra-base-discriminator` | `1ae76a97c7e8` | ❌ RISK (pickle.malformed) | ✅ COMPLIANT |
| `google/vit-base-patch16-224` | `3f49326eb077` | COMPLIANT | ✅ COMPLIANT |
| `hexgrad/Kokoro-82M` | `f3ff3571791e` | ❌ RISK (pickle.unanalyzable) | ✅ COMPLIANT |
| `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | `cf031e020336` | ❌ RISK (findings.truncated, pickle.unanalyzable) | ✅ COMPLIANT |
| `nomic-ai/nomic-embed-text-v1.5` | `e9b6763023c6` | COMPLIANT | ✅ COMPLIANT |
| `openai-community/gpt2` | `607a30d783df` | ❌ RISK (pickle.malformed) | ✅ COMPLIANT |
| `openai/clip-vit-base-patch32` | `3d74acf9a28c` | ❌ RISK (pickle.unanalyzable) | ✅ COMPLIANT |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | ❌ RISK (pickle.malformed, pickle.unanalyzable) | ✅ COMPLIANT |
| `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf54` | ❌ RISK (pickle.malformed, pickle.unanalyzable) | ✅ COMPLIANT |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b` | ❌ RISK (pickle.unanalyzable) | ✅ COMPLIANT |
| `timm/mobilenetv3_small_100.lamb_in1k` | `1824797e7887` | COMPLIANT | ✅ COMPLIANT |

## False-positive causes, before the fixes

| Model | Finding | Cause |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | pickle.unanalyzable | tensor storage read as an archive / too large |
| `google-bert/bert-base-uncased` | pickle.malformed | tensor storage read as a pickle |
| `google-t5/t5-small` | pickle.malformed | tensor storage read as a pickle |
| `google/electra-base-discriminator` | pickle.malformed | tensor storage read as a pickle |
| `hexgrad/Kokoro-82M` | pickle.unanalyzable | tensor storage read as an archive / too large |
| `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | findings.truncated, pickle.unanalyzable | many storage findings (truncation summary); tensor storage read as an archive / too large |
| `openai-community/gpt2` | pickle.malformed | tensor storage read as a pickle |
| `openai/clip-vit-base-patch32` | pickle.unanalyzable | tensor storage read as an archive / too large |
| `sentence-transformers/all-MiniLM-L6-v2` | pickle.malformed, pickle.unanalyzable | tensor storage read as a pickle; tensor storage read as an archive / too large |
| `sentence-transformers/all-mpnet-base-v2` | pickle.malformed, pickle.unanalyzable | tensor storage read as a pickle; tensor storage read as an archive / too large |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | pickle.unanalyzable | tensor storage read as an archive / too large |
