# Clinical Association and Reliability of Classical and Advanced P300 Waveform Features in Alzheimer's and Parkinson's Disease

Master's thesis, Master of Science in Engineering, FH Technikum Wien — Medical Engineering & eHealth.

Goran Nikic · Vienna, October 2026

---

## What is in this repository

| | |
|---|---|
| **`code/`** | The complete processing and analysis pipeline. It has **its own README** ([`code/README.md`](code/README.md)) explaining how the pipeline was used for this thesis and how it can be run again. |
| **`results/`** | The computed output of every analysis, as CSV: the cross-dataset comparisons of each feature, the reliability estimates, the association and classification analyses, the power calculations and the robustness checks. These are the tables Chapter 3 of the thesis reports. |
| **`Abstract_….pdf`** | Title page, German abstract (Kurzfassung), English abstract and table of contents. |

## The data are not in this repository

**No EEG recordings, no participant tables and no per-participant feature
tables are included here.** What is published is code and group-level result
tables only. The pipeline takes the recordings as input, so it cannot be run
from this repository alone — you need to obtain the data first.

### Two of the three cohorts are public

**Parkinson's disease cohort.** The EEG recordings come from the Patient
Repository for EEG Data + Computational Tools (PRED+CT) at the University of
New Mexico, dataset d001 "Parkinson's Oddball", contributed by James F.
Cavanagh and released under a Public Domain Dedication and Licence (PDDL) v1.0:

- https://predict.cs.unm.edu/downloads.php

The participant table, which carries group, age, sex and the medication state
of each session, comes from the authors' own deposit of the same recordings in
BIDS form on OpenNeuro:

- https://openneuro.org/datasets/ds003490 — doi:10.18112/openneuro.ds003490.v1.1.0

**Healthy reference cohort.** An auditory oddball dataset on OpenNeuro:

- https://openneuro.org/datasets/ds003061

Point the pipeline at these two and it will reproduce their feature tables and
every result that does not depend on the Alzheimer's cohort.

### The third cohort is not public

The Alzheimer's cohort was provided by The Siesta Group for this thesis under
a written agreement and may not be redistributed. It consists of pseudonymised
clinical recordings, and neither the recordings nor any table derived from
them at the level of individual participants is published — here or anywhere
else. Every result involving that cohort appears only as a group statistic.

## Why only the abstract, and not the whole thesis

Because the Alzheimer's cohort cannot be published, the thesis is subject to a
restriction notice and is not being released in full. What is published here
is the front matter: the title page, the German and English abstracts and the
table of contents, so the aim, the approach and the structure of the work can
be read and cited.

## Reusing the pipeline

Start with [`code/README.md`](code/README.md). It describes the architecture,
how each cohort is handled, and the commands that produced the tables in
`results/`.

The pipeline is built so that one shared core measures every cohort
identically — only file format, event decoding and artifact handling differ
per cohort — which is what makes the cross-cohort comparison meaningful in the
first place.
