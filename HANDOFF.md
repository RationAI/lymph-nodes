# Handoff: register H&E to CK-IHC sentinel lymph node slides, to transfer IHC-model labels to H&E
State as of 2026-11-18 · branch `feature/he-registration` (from `feature/prop/train` 4d10d7d) · worktree `~/Documents/RationAI/lymph-nodes-he-registration`
(Do not trust numbers you cannot trace to a file listed here. This session starts fresh: the
other session's notes are in `~/Documents/RationAI/lymph-nodes/HANDOFF.md` — training — and
`PAPER_HANDOFF.md` there — paper.)

## 1. Goal
The project trains a detector of isolated tumour cells (ITCs) and metastases in
cytokeratin (AE1/AE3, H-DAB) IHC sentinel lymph nodes (breast cancer), without any positive
lymph node in training. The long-term goal is an **H&E** model, for which ITC labels are
nearly impossible to annotate. Plan: once the IHC model is good enough, use it as a
ground-truth extractor — run it on IHC slides and carry its cell-level output onto the
**same section restained with H&E** by image registration. This branch builds that
registration and, first, finds out how many such pairs exist.

## 2. Data (facts measured 2026-10-29, files in the main worktree)
- Slides: H&E `/mnt/data/MOU/lymph_nodes/dataset1-he-2023/SNB_HE_CASE_<c>_SLIDE_<s>.mrxs`;
  IHC 2023 = `mmci_ihc_2023` (same naming with `SNB_IHC_`). On kubas-cluster only
  (sensitive; never copy WSIs off it).
- Pairing table: `~/Documents/RationAI/lymph-nodes/analysis/reports/he_ihc_pairs/blocks.csv`
  (one row per IHC slide: `verdict`, matched H&E `block`, labels `label` (IHC) /
  `block_label` (H&E), outline overlap `section_dice`, coarse `same_section`), plus
  `slides.csv`, `pieces.csv` and contact sheets (`sheet_same_section.jpg`, ...). Made by
  `~/Documents/RationAI/lymph-nodes/analysis/he_ihc_pairs.py` (node-shape + haematoxylin
  pattern matching within a case). Both files are untracked there: copy them in here if
  you build on them.
- **Names don't pair**: IHC `CASE_c_SLIDE_s` is often another H&E slide of the case
  (107 namesake, 73 other of 180 found); always use `blocks.csv`.
- 333 IHC slides: 150 matched, 19 probable, 5 matched to several H&E, 6 outline only,
  148 no match, 4 cases without H&E. **29 IHC-positive**; **18 of them have an H&E match**
  (16 matched + 2 probable): 11 positive on both, **7 positive on IHC only** (ITCs H&E
  doesn't show — the most valuable for transfer).
- Same section restained vs neighbouring section: the coarse call (`same_section`,
  outline Dice) flags only 3 pairs (cases 19_1, 80_1, 123_2), all negative; outline Dice
  0.95 (same) vs 0.55 (rest, median). Visually identical outlines were also seen for
  cases 134_1/134_2 and 70_7. Among the 18 positive pairs, 134_2 (0.91), 134_1 (0.86) and
  perhaps 2_1 (0.79) look like restains; the rest (0.3-0.6) are probably neighbouring
  sections, where cell-level transfer is invalid (an ITC may not be in the other cut).
- H&E quality: 48 H&E scans badly blurred (sharpness < 5), ~70 soft (< 50) (`slides.csv`).
- Other cohorts: FN Brno has AE1/AE3 + H&E **serial** pairs (adjacent sections, CZI,
  0.11 um/px; 814 CZI, 105 cases) — region-level at best, not cell-level.

## 3. Existing code to reuse
- `~/Documents/RationAI/image-registration` (master): TMA-core registration tool (legacy
  `imreg`, pdm; `TMA_RATIOPATH_IMPLEMENTATION_PLAN.md`) — aligns DAB-derived epithelium
  masks onto H&E TMA cores. Check what transfers to whole lymph node sections.
- This repo: slide/tiling infrastructure (`preprocessing/`, `configs/`, `ratiopath`,
  `rationai-mlkit`), MLflow tracking, kube-jobs launchers (`scripts/`), the IHC model's
  prediction masks (`lymph_nodes/callbacks/prediction_masks.py`, 3.4 um, logged per
  validation run) and the refined DAB labels (`preprocessing/refined_annotation_masks.py`).

## 4. Results so far
- `scripts/registration/restain_score.py` (step 1's scorer; method in its docstring):
  coarse rigid search (4 orientations, -12..12 deg, masked NCC at 62/31 um/px), up to 36
  patches each located locally (1.9 um/px, +-4 deg), refined at 0.49 um/px, nuclei
  (haematoxylin peaks) matched within 2 um against chance (IHC nuclei shifted 10 um) ->
  kappa; global RANSAC similarity over patches -> residual um. `pairs.csv` = 187 pairs
  (180 IHC slides: every `block` H&E and the `section` H&E of found pairs; names and
  verdict only). Launcher `restain_job.py` (`--pilot` = 10 pairs).
- Synthetic check only (no patient data; scratchpad `test_restain_synthetic.py`, 4 x 3 mm,
  12 patches): restain kappa 0.95 / residual 0.01 um / inliers 1.00, transform recovered
  (turned, 181.3 deg); neighbour (45% shared nuclei + smooth 12 um deformation) kappa 0.16 /
  14.6 um / 0.25; mirrored neighbour 0.15 / 12.8 um. ~5 s per pair. Real thresholds still
  to be set on the pilot.
- **Pilot v1** (MLflow run 7ec93f31, commit b4d6fe1, job `ln-he-restain-score-pilot-pekar`,
  10 pairs, ~55 s each): kappa median 0.01-0.03 for **every** pair, incl. the coarse
  same-section ones (19_1, 80_1, 123_2: outline Dice 0.89-0.96) and 134_1/134_2/2_1;
  residuals 38-766 um. The best patch per pair reaches only kappa 0.07-0.11. QC sheets
  (19_1, 134_2): structures 20-50 um across (vessels, fat, fibrin, pale stroma) sit at the
  same place within ~5-10 um, but the nuclei don't correspond.
  Bug found: `match_template` gave NCC up to 5945 in near-blank windows (glass), so some
  patches locked onto background (inflates residuals; doesn't explain the dense
  lymphoid patches). Fixed (commit pending): blank windows dropped (`window_std`), nuclei
  also matched over a +-6 um search (`*_searched`, chance searched alike), `--self`
  positive controls (IHC vs itself), a 480 um context view in the QC sheets. Synthetic:
  self kappa 1.00, restain searched 0.92, neighbour 0.14.
  Reading so far (to confirm with pilot v2): the "same-section" pairs look like
  **neighbouring sections** of a ribbon (same structures, other nuclei), not the same glass
  restained.

## 5. Decisions & rationale
- Cell-level transfer only for **the same section restained**; neighbouring sections give
  at most region-level labels (macro/micrometastases), never ITC labels.
- Accuracy target: ITCs are ~8-13 um across, so the registration must be good to ~2-3 um
  over the node (to be confirmed with measured residuals).

## 6. Open problems
- How many pairs are true restains, especially among the 18 positive ones? (Possibly only
  2-3.) If that few, the bottleneck is data: ask the lab to restain more positive IHC
  slides with H&E (a decision for the user, not code).
- Which order were sections stained (IHC first, then destained to H&E, or the reverse)?
  Ask the user / lab; it affects tissue damage and colour.

## 7. Next steps
1. **Count the true restains** at cell level. For every matched/probable pair: rigid
   (tissue outline) then non-rigid registration at a high resolution; detect nuclei on
   both (haematoxylin channel) and measure the share of IHC nuclei with an H&E nucleus
   within ~3 um. Same section -> most match; neighbouring sections (>= ~3-4 um apart) ->
   few. Done when: a per-pair score with a threshold checked visually on high/low
   examples (xOpat overlays), and the count of restained positive pairs reported to the
   user.
2. **Go / no-go with the user** on the counts (and on asking the lab for more restains).
3. **Registration method**: start from the simplest that meets the accuracy target.
   Before choosing, run the `lit-search` skill on H&E-IHC whole-slide registration
   (candidates to verify, not claims: VALIS, the ACROBAT challenge on breast H&E-IHC,
   DeeperHistReg, ANHIR) and compare with the legacy `imreg`. Done when: residual error
   (um) measured on landmarks/nuclei of a few restained pairs.
4. **Label transfer**: warp the IHC model's prediction masks (or the refined DAB labels)
   into H&E space; QC gallery of transferred lesions. Done when: the transferred labels of
   the restained positive pairs reviewed with the user.

## 8. Gotchas (from the main branch)
- Jobs: `kube_jobs.submit_job` via launchers in `scripts/` (`uv run --group run ...`);
  names get `-pekar`, keep <= 57 chars; code must be pushed (jobs clone the branch);
  gated HF models need `HF_TOKEN` in the environment (never hard-code it).
- kubas-cluster namespace memory quota is often full: pods wait ("exceeded quota").
  `mig-2g.20gb` (A100 node) has been unallocatable since ~2026-11-15: use `mig-2g.24gb`.
- Never build datasets on the Jupyter pod; never move data between machines without asking.
- `uv run` locally needs `--with openslide-bin` to import `lymph_nodes` (no OpenSlide dylib).

## Changelog
- 2026-11-18: branch and worktree created; facts on the pairs collected from
  `analysis/reports/he_ihc_pairs/blocks.csv`.
- 2026-11-18 evening: restain scorer + pair list + launcher written, synthetic check passed;
  committed b4d6fe1 and pushed (user's go); pilot v1 7ec93f31: no pair restained at cell
  level, a blank-window NCC bug; fixes + self controls written (uncommitted).
