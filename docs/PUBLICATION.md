# Publishing the Artifact

Suggested repository name: `enhanced-projection-decoding`.
Upload the contents of this directory as the repository root, excluding
`outputs/`, virtual environments, and Python caches. Keep `reference/` so that
readers can regenerate the reported table without rerunning the timed benchmark.
Choose a software license with the coauthors and add it as `LICENSE`.

After creating the GitHub repository, replace `OWNER` below with its actual
owner. This is a URL template, not an already published repository:

```text
https://github.com/OWNER/enhanced-projection-decoding
```

Create a tagged release matching the submitted manuscript and record its commit
in the final artifact. An archived release, for example through Zenodo, can
provide a persistent identifier. Add the final paper identifier when available.

## Placement in the Paper

Place one short availability sentence at the end of the introductory paragraph
of the finite-length evaluation in the Lee subsection, before the parameter
and measurement details. This is where readers first encounter the experiments.
It can point to the code and reference data without adding another paragraph
heading or method abbreviation:

```latex
The experimental code and reference data are available at
\url{https://github.com/OWNER/enhanced-projection-decoding}.
```

If wider visibility is desired, put the link in a footnote at the first mention
of the experimental evaluation in the introduction, and refer to that artifact
in the experiment section without repeating the URL. Avoid placing a long URL
in the abstract. Use the venue's anonymous-artifact mechanism during anonymous
review, if required. The placeholder should only enter the manuscript after
the repository address has been confirmed.
