# Data collection and validation protocol

Version 0.2 — research use only

This protocol freezes the intended acquisition procedure for the current software. It
does not constitute ethics approval, medical-device authorization, or a clinical study
protocol. A movement-disorder clinician and the responsible ethics/privacy body must
approve the final study documents before participant recruitment.

## 1. Intended use

The system measures fine-motor behavior during spiral drawing and produces an
experimental pattern-similarity score. It is intended for supervised research into
Parkinson's screening. It must not diagnose, rule out, grade, or monitor Parkinson's
disease in clinical care.

The current models were developed from small case/control archives. Their outputs are
not population-calibrated probabilities. The 0.75 elevated-signal boundary is a
conservative engineering decision rule, not 75% diagnostic confidence or a clinically
validated referral threshold. Out-of-domain voice recordings must not receive a score.

## 2. Equipment and software

- Windows computer running Python 3.11–3.13
- pen display or drawing tablet with current manufacturer driver
- Windows Ink enabled so Pointer Events expose `pointerType=pen`, pressure, and tilt
- stable tablet placement; browser zoom at 100%; no remote-desktop input
- project version and model artifact checksum recorded for every study batch

Record tablet make/model, active-area dimensions, driver version, operating-system
version, browser version, display scaling, and whether the tablet has an integrated
screen. Do not combine devices in analysis without explicitly modeling or externally
testing device effects.

## 3. Participant metadata and labels

Use a pseudonymous participant code. Keep the re-identification key outside this
application in an approved study system. The application records:

- dominant hand
- broad age band
- Parkinson's medication timing: not applicable, before dose/OFF, after dose/ON, or unknown
- trial hand, task mode, repetition, canvas size, capture time, and device input type

For a real study, collect clinician-reviewed labels separately. At minimum include:

- neurologist-confirmed Parkinson's disease with diagnostic criteria and disease duration
- healthy age-matched controls
- essential tremor
- drug-induced or vascular parkinsonism and other available movement-disorder mimics
- relevant hand injury, arthritis, neuropathy, stroke, vision problems, and medications

The clinician assigning the reference label should not see the model score.

## 4. Standardized acquisition

1. Confirm the participant understands the research task and has provided approved consent.
2. Seat the participant comfortably with the tablet centered and stationary.
3. Explain that this is not a diagnostic examination and that they should draw naturally.
4. Enter only the pseudonymous code and required metadata.
5. Let the participant use **Clear and retry** once to become comfortable if necessary;
   do not export the abandoned attempt.
6. Complete the two application-guided spiral trials in the fixed order:
   dominant-hand static, then dominant-hand blinking. Non-dominant-hand collection is
   optional research data and must not be mixed into the primary dominant-hand score.
7. If voice collection is approved, record three six-second sustained /a/ phonations
   using the same microphone position and quiet-room conditions. Retain derived features
   only unless the consent and data-governance protocol explicitly permits raw audio.
8. Start at the green center and trace outward in one continuous movement. Do not coach
   speed after drawing begins.
9. If the application rejects a trial, repeat it once and record the rejection reason in
   the study log. Repeated failures should be retained as a feasibility outcome rather
   than silently excluded from the cohort.
10. Export the JSON into the approved study location, verify receipt, and delete the local
   session if required by the study's retention policy.

Use the same instructions, dominant-hand definition, browser zoom, tablet placement, and room
conditions for cases and controls. Collection staff should not change technique based on
known diagnosis.

## 5. Captured point schema

Each exported trial contains the following raw point fields:

| Field | Meaning | Unit |
|---|---|---|
| `x`, `y` | Position in the fixed drawing canvas | canvas pixels |
| `t` | Time since first recorded point | milliseconds |
| `pressure` | Browser-normalized pen pressure | 0–1 |
| `tilt_x`, `tilt_y` | Stylus tilt from the surface-normal axes | degrees |
| `pointer_type` | `pen`, `touch`, `mouse`, or `unknown` | category |
| `stroke` | Sequential contact/stroke identifier | integer |

The session also contains the task mode, hand, repetition, canvas dimensions, automated
quality output, timestamps, and pseudonymous metadata. Mouse and touch captures are
flagged as demonstration-only and should be excluded from tablet-model validation.

## 6. Automated quality checks

A trial is rejected from scoring when it has fewer than 80 samples, lasts under one
second or over 90 seconds, covers less than 25% of the canvas dimension, or contains
fewer than approximately 1.8 turns. More than five detected turns generates a warning.

These are engineering safeguards, not validated clinical quality thresholds. Analyze
failure rates by diagnosis, age, device, hand, and site: a quality rule that rejects one
group more often can introduce selection bias.

## 7. Dataset construction

- Freeze raw exports as immutable source data; derive processed features reproducibly.
- Assign a stable participant identifier across visits.
- Split by participant before preprocessing, augmentation, feature selection, or tuning.
- Keep all hands, repetitions, visits, and augmented versions of one participant in the
  same split.
- Preserve natural class prevalence for the intended-use validation cohort or adjust
  calibration explicitly; do not interpret case/control sampling as prevalence.
- Maintain a data manifest with checksums, consent/retention status, site, device, label
  source, and exclusion reason.

## 8. Validation design

Development should use grouped nested cross-validation. The final test must be an
untouched external cohort from a different site and, preferably, a different compatible
tablet. Pre-register the primary endpoint, threshold-selection rule, missing-data rules,
and subgroup analyses before opening that external set.

Report at minimum:

- participant count and prevalence with a recruitment flow diagram
- sensitivity, specificity, ROC-AUC, PR-AUC, and 95% confidence intervals
- positive and negative predictive values at the intended-use prevalence
- calibration intercept/slope, reliability plot, and Brier score
- invalid-capture and missing-data rates
- results by site, tablet, age band, sex, hand, disease stage, and medication state
- Parkinson's versus essential tremor/mimics, not only Parkinson's versus healthy controls
- performance of spiral-only, metadata-only, and combined models

After threshold selection, validate prospectively without updating the model. A clinical
workflow evaluation must measure referral consequences, false reassurance, false alarms,
usability, and accessibility—not only discrimination metrics.

## 9. Model update and release gates

Do not replace the model artifact merely because a new model has a higher internal AUC.
Require versioned training data, reproducible code, leakage review, subgroup review,
external performance, calibration, and clinician sign-off. Keep old artifacts and model
cards so every historical score can be reproduced.

Public or clinical deployment remains blocked until the intended-use validation,
privacy/security assessment, regulatory review where applicable, and clinician-owned
referral pathway are complete.
