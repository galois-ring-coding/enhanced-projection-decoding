# Publishing This Release

Repository: https://github.com/galois-ring-coding/enhanced-projection-decoding

This trimmed package includes the current manuscript's three experiment
groups and the five earlier provenance files listed in README.md.
Obsolete measurements, pilot runs, former supplements, and unrelated
Hamming/rank experiments are omitted. Shared source dependencies are retained
without changing the scientific code or its recorded hashes.

Publish the current source, tests, documentation, requirements, and
`reference/` at the repository root. Upload the complete ZIP as a release
asset: one scale JSONL file exceeds GitHub's ordinary 100 MiB file limit.
Extracting the ZIP supplies the layout used by the commands in README.md.

Before publication, run the documented analyses and checks, regenerate
`reference/sha256.json`, and regenerate `RELEASE_MANIFEST.json` after all
outputs are finalized. The latter contains a relative path, byte count,
and SHA-256 for every included file except itself.
The ZIP's adjacent `.sha256` file identifies the exact archive.

Use a tag corresponding to the submitted manuscript, for example
`ccp-manuscript-20261122`, and record the actual commit and asset checksum.
A locally prepared ZIP is not a published release.

As checked on 2026-11-22, the public default branch lacks the current scale
and MMT experiments. The local environment has no GitHub login available,
so this package has not been uploaded. Publication must finish before
claiming that all current results are publicly reproducible.
