# JOSS submission checklist

Working notes for submitting $N^2$ to the [Journal of Open Source Software](https://joss.theoj.org).
Delete this file (or move it out of `paper/`) once the submission is accepted — it is not
part of the paper.

Reference: <https://joss.readthedocs.io/en/latest/submitting.html>

---

## ⚠️ Blocking — must be resolved by a human before submitting

- [ ] **Missing ORCIDs.** `paper/paper.md` and `CITATION.cff` have no ORCID for **Caleb
      Chin**, **Harshvardhan Maskara**, or **Anish Agarwal**. JOSS validates ORCIDs, so a
      placeholder like `0000-0000-0000-0000` (which was in the old draft on the `joss`
      branch) will fail. Either supply the real IDs or leave the field out entirely.
      Also confirm the seven ORCIDs that *are* listed belong to the right people.

- [ ] **Confirm the author list and order.** The current list matches the earlier JOSS
      draft. Note two discrepancies to resolve:
      - The newer N$^2$-Bench manuscript (`~/Automated_NN-2/main.tex`) additionally lists
        **Dwaipayan Saha** (Columbia). Decide whether they belong on the JOSS paper.
      - `pyproject.toml` lists a different set again (no Anish Agarwal, no Dwaipayan Saha).
      Whatever is decided, make `paper.md`, `CITATION.cff`, and `pyproject.toml` agree.

- [ ] **Confirm the corresponding author.** Currently only Jacob Feitelberg is marked
      `corresponding: true`, and Caleb Chin / Aashish Khubchandani are marked
      `equal-contrib: true`. The old draft marked five authors as corresponding.

- [ ] **Sign off on the AI usage disclosure.** JOSS now *requires* this section. The text
      in `paper.md` discloses that generative AI assisted in drafting the paper and the
      user-facing documentation, under author direction and with author verification.
      An HTML comment right below it flags what to check. **Every co-author must confirm
      it is complete** — in particular, whether generative AI was used anywhere in the
      library source, tests, or experiment scripts. Amend the paragraph accordingly.

- [ ] **Verify the research-impact claims.** The "Research impact" section states that
      NN variants often outperform classical methods on N$^2$-Bench and that
      distributional NN beats every scalar method on HeartSteps. Confirm these match the
      final numbers in the benchmark manuscript before submitting.

- [ ] **Decide what to do about `NadarayaWatsonEstimator`.** It cannot be instantiated
      (missing `_calculate_distances`), so it is currently excluded from the public API
      and the docs. Either finish it or delete `src/nsquared/nadaraya_watson.py`. A
      reviewer reading the source will notice dead code.

---

## Reconciling with the existing `joss` branch

There is already a remote branch `origin/joss` containing an earlier, partial
`paper/main.md` (empty "Unified framework" and "Test bench" sections, no State of the
field / Software design / Research impact / AI usage sections) plus a 115-entry
`paper/ref.bib`. That branch has **also diverged from `main` in the source tree** — most
notably it renames `aw_nn` → `star_nn` and rewrites `estimation_methods.py` and
`fit_methods.py`.

The work on this branch was written fresh against `main` and salvages the useful parts of
that draft (author list, ORCIDs, acknowledgements, the summary and statement-of-need
material). Decide whether `origin/joss` should be merged, rebased, or abandoned before
submitting, so there is a single source of truth.

---

## Release and archive (do this immediately before submitting)

JOSS requires a tagged release archived with a DOI.

- [ ] Bump the version in `pyproject.toml` and in `CITATION.cff` (`version:` and
      `date-released:`). Follow [`RELEASE.md`](../RELEASE.md).
- [ ] Tag and publish the release on GitHub. The `publish.yml` workflow pushes it to PyPI.
- [ ] Archive the release with [Zenodo](https://zenodo.org) (or figshare) to mint a DOI.
      The metadata — title, author list — must match `paper.md`.
- [ ] Record the version number and DOI; both go in the submission form.
- [ ] Add the DOI to `CITATION.cff` and to the README badge row.

---

## Submission form

Submit at <https://joss.theoj.org/papers/new>. You will need:

| Field | Value |
| --- | --- |
| Repository URL | `https://github.com/aashish-khub/NearestNeighbors` |
| Software version | the tag created above |
| Archive DOI | the Zenodo DOI from above |
| Branch (if not default) | leave blank if the paper is merged to `main` |

`paper.md` and `paper.bib` must live in the same directory, which they do (`paper/`).

---

## Review criteria — current status

Checked against <https://joss.readthedocs.io/en/latest/review_criteria.html>.

### Software

| Criterion | Status |
| --- | --- |
| OSI-approved license, as an actual file | ✅ `LICENSE` (MIT) |
| Open, browsable repository, public issue tracker | ✅ |
| More than six months of public history | ✅ first commit 2024-12-05, 458 commits |
| Multiple contributors | ✅ seven |
| Tagged releases | ✅ `v1.0.0`, `v1.1.0`; a JOSS release still needs tagging |
| Packaged to language standards | ✅ on PyPI as `nsquared` |
| Substantial scholarly effort | ✅ ~4,000 lines of library code plus the benchmark |

### Documentation

| Criterion | Status |
| --- | --- |
| Statement of need | ✅ README "Why $N^2$?"; `paper.md` |
| Installation instructions incl. dependencies | ✅ README; [`docs/installation.md`](../docs/installation.md) |
| Example usage | ✅ README quickstart; [`docs/quickstart.md`](../docs/quickstart.md); `examples/` |
| API documentation | ✅ [`docs/api_reference.md`](../docs/api_reference.md) plus docstrings |
| Automated tests | ✅ 116 tests, run in CI on Python 3.10–3.12 and on macOS/Windows |
| Community guidelines (contribute / report / support) | ✅ [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md), issue templates |

### Paper

| Criterion | Status |
| --- | --- |
| 750–1750 words | ✅ ~1,240 |
| Summary for a non-specialist | ✅ |
| Statement of need | ✅ |
| State of the field | ✅ compares `scikit-learn`, `fancyimpute`, `syntheticNN` |
| Software design | ✅ |
| Research impact | ⚠️ written; claims need author verification (above) |
| AI usage disclosure | ⚠️ written; needs co-author sign-off (above) |
| References with DOIs, full venue names | ✅ `paper.bib` |
| Authors and affiliations with ROR | ✅ ROR IDs included; three ORCIDs missing |

---

## Optional improvements

Not required for acceptance, but reviewers often note them:

- [ ] Publish the `docs/` tree as a rendered site (GitHub Pages via `mkdocs` or Sphinx)
      rather than Markdown files in the repo.
- [ ] Add a test coverage badge (e.g. Codecov).
- [ ] Add a `CHANGELOG.md`.
- [ ] Register the `earnings` dataset loader — `src/nsquared/datasets/earnings/loader.py`
      has no `@register_dataset` decorator, so it is unreachable through
      `NNData.create()` and absent from `get_available_datasets()`.
- [ ] Implement `SyntheticDataLoader._make_mnar` — the `mode="mnar"` option currently
      raises `NotImplementedError` (pinned by `tests/test_datasets.py`).
