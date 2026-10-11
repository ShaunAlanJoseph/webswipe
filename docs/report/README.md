# WebSwipe project documentation

This is the submission report for the selected app after the **11 October 2026 afternoon left-pinch update**. Open `WebSwipe_Project_Documentation.pdf`; edit `elsarticle-template.tex` and `mybibfile.bib`. The main PDF now includes a linked reproducibility appendix containing the version manifest, selected control parameters, investigated variants, normalization derivation, and numerical audit. The same content is also built as `WebSwipe_Reproducibility_Supplement.pdf`; both use `reproducibility_body.tex`. This documentation package is not the runnable demo package.

The professor's supplied template is retained: Elsevier `elsarticle` preprint, 12 pt, double spacing, line numbers, numerical references, and the five required main sections. The report discusses 15 works from 2025 and cites the foundational methods separately. Author names and identifiers are unchanged.

## Build and check

With TeX Live installed:

```sh
sh build.sh
```

`build.sh` runs PDFLaTeX, BibTeX, and two resolving PDFLaTeX passes. It creates `elsarticle-template.pdf` and the identical `WebSwipe_Project_Documentation.pdf`, then compiles `WebSwipe_Reproducibility_Supplement.tex` twice. The source ZIP can also be imported into Overleaf; select `elsarticle-template.tex` as the main file, or the supplement's `.tex` file to build the supplement.

To rebuild the six plotted figures, install NumPy and Matplotlib, then run:

```sh
python3 generate_figures.py
```

The main report uses five plotted figures and a recorded-frame interface illustration. The large image, `images/app_replay_standing.png`, uses frame 900 of the standing recording with the app keyboard and display-toggle hint drawn over it; no detection or performance measurement is shown. The archived input is `evidence/standing_interface_frame.png`; `render_standing_interface.py` reproduces the overlay using the frozen drawing helpers. Both enlarged details remain viewports into the earlier `images/app_replay.png` interface replay, with preloaded candidates and a placeholder timing field. The three pinch photographs are unchanged. Frame and crop provenance is in `evidence/figure_provenance.json`. The supporting `images/webcam_comparison.pdf` is retained outside the main report.

## Evidence and reproducibility

`evidence/` retains both historical results and the selected version's results. In particular:

- `selected_capture.json` and `selected_decoder_scores.json` support the current capture/word tables.
- `review_numerical_audit.json` records the subject-9, frame-rate, retry-count, and noise-grid checks. `d1_prior_s*.json` supports the tuned baseline means and standard deviations.
- `lighting_training_report.json` and `lighting_training_args.yaml` identify the final detector continuation.
- `quick_pinch_live_diagnosis.json`, `quick_pinch_live_replay.json`, and `quick_pinch_app_parity.json` document the latest control fix.
- `runtime_snapshot.json` records model hashes, source hashes, portable app arguments, and pinned package versions.
- `pinch_case.json` contains the observations used in the failure/recovery plot.
- `control_validation_history.tex` preserves the detailed sequence of control fixes and revision-specific checks; `synthetic_experiment_notes.tex` preserves the fuller synthetic results. These are supporting text fragments, not separate build targets or sections included in the main PDF.

The selected difficult-lighting replay reaches 15/15 complete capture, 11/15 top-1, and 14/15 top-4. The original 29-attempt headline results remain 23/29, 21/29, and 22/29. The older development set's selected scores are 38/45 top-1 and 42/45 top-4, so the report does not claim that every earlier score was preserved. The final control replay recovers both missed brief pinches, but a fresh live confirmation of that last revision is still outstanding.

The report separates simulations, recorded headset tests, recorded webcam regression checks, and live diagnostic observations. It does not report invented WPM, independent multi-user accuracy, or end-to-end latency. Software test counts are tied to revisions and are not added together.

The local copies of `elsarticle`, `algorithms`, and `algorithmicx` retain their upstream source and notices in `vendor-source/`. Their presence makes the supplied template easier to compile on systems where those packages are missing.

- `evidence/diagnostic_detail_notes.tex` retains the detailed session timing, cadence checks, retry history, and secondary results shortened in the submission report. It is an archival text fragment, not a build target.
