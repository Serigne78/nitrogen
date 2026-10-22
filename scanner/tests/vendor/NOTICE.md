# Vendored third-party test corpora

These files are scanned by the neutral benchmark (`scripts/benchmark_neutral.py`,
ADR 0007). They are **never loaded or executed** by Nitrogen — only their
opcodes are read. They are redistributed here under their original licences.

## picklescan/ — from mmaitre314/picklescan
- Source: https://github.com/mmaitre314/picklescan, `tests/data/`
- Commit: 5e7ab22cb5bc59c4276820df5153829db4c751e8
- Licence: MIT (see picklescan/LICENSE)

These are picklescan's own malicious/benign pickle test fixtures. Ground truth:
a file whose name starts with `malicious`, `bad`, `broken` or ends with
`sploit` is expected to be flagged; a file starting with `benign` is expected
to pass.
