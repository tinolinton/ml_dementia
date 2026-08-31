# Study Plan — Logistic Regression Model for Dementia Prediction in Zimbabwe

**Project code:** `ml_dementia`
**Version:** 0.1 (planning draft)
**Date:** 2026-08-25
**Status:** Pre-data-collection / methodological development phase
**Primary deliverables:** `plan.md` (this document), `data.sql` (calibrated synthetic dataset with engineered data-quality defects)

---

## 1. Project Overview

### 1.1 Aim

Develop and internally validate a **logistic regression (LR) classification model** that estimates an individual's probability of dementia among adults aged ≥60 in Zimbabwe, using routinely collectable field variables (sociodemographic, vascular/metabolic, lifestyle, psychosocial, sensory, and HIV-related).

### 1.2 Research questions

1. Which of a parsimonious set of community-measurable risk factors are independently associated with dementia status in older Zimbabweans?
2. Can these predictors yield a screening/risk-stratification tool with acceptable discrimination (target AUC ≥ 0.75, sensitivity ≥ 0.80 at a clinically motivated operating point) for use by primary-level health workers?
3. How do model performance and calibration vary across key subgroups (sex, urban/rural residence)?

### 1.3 Hypotheses (directional)

- H1: Older age, fewer years of education, hypertension, diabetes, prior stroke, depressive symptoms, hearing/vision impairment, physical inactivity, social isolation, and HIV infection are each positively associated with dementia odds.
- H2: An LR model using only interview-collected variables achieves AUC > 0.75 without neuroimaging or biomarkers.
- H3: Discrimination is maintained but calibration degrades across urban/rural strata.

---

## 2. Literature Review — State of Knowledge

> Note on sourcing: figures below were cross-checked against the WHO dementia fact sheet (retrieved Aug 2026) and a PubMed literature scan (`dementia prevalence Zimbabwe`, retrieved Aug 2026). Where exact figures could not be re-verified online during planning, they are presented as approximate ranges from memory of the published literature and flagged accordingly.

### 2.1 Global state

- **Burden:** ~57 million people living with dementia worldwide as of 2021, with nearly 10 million new cases per year; >60% live in low- and middle-income countries (LMICs); global cost estimated at US$1.3 trillion (2019) — _WHO Dementia Fact Sheet, 2025 update_.
- **Projection:** ADI/WHO projections converge on roughly a doubling-plus of cases by 2050 (~139–152 million), with the steepest relative growth in LMICs (_World Alzheimer Report 2021_; GBD 2019/2021 Dementia Forecasting Collaborators, _Lancet Public Health_ 2022).
- **Prevention framework:** The Lancet Commission on dementia (Livingston et al., 2020; 2024 update) attributes ~45% of dementia risk to **14 modifiable factors** across the life course: low education, hearing loss, hypertension, smoking, obesity, depression, physical inactivity, diabetes, social isolation, excess alcohol, traumatic brain injury, air pollution (2020), plus vision loss and high LDL cholesterol (2024).
- **Risk-prediction modelling:** Large cohorts (Framingham, UK Biobank, ADNI, Rotterdam Study) support multivariable risk scores; logistic regression remains a strong, interpretable baseline that frequently matches more complex ML on tabular clinical data (AUC typically ~0.72–0.88 depending on predictor richness and follow-up horizon). Established instruments include the Framingham General Dementia Risk Score and ANU-ADRI — none calibrated for African populations.
- **Known limitation:** nearly all global prediction models are trained on high-income, majority-ancestry cohorts; external validation in African settings is essentially absent.

### 2.2 Continental state (Africa)

- **Prevalence:** systematic reviews report pooled dementia prevalence among adults ≥65 in Sub-Saharan Africa of roughly **6–10%**, with very wide between-study heterogeneity (Guerchet et al., 2016; Longdon et al., 2020). Some early studies using unadapted Western criteria reported implausibly low or high rates.
- **Diagnostic infrastructure:** the **10/66 Dementia Research Group** protocols (Prince et al., 2007) and the CSI-D provide validated, education-fair case-ascertainment algorithms for LMICs; the **IDEA** study (Tanzania) compared diagnostic criteria head-to-head and showed substantial disagreement between DSM/ICD and 10/66 algorithms.
- **Cultural framing:** dementia is frequently normalised as "ageing" or attributed to spiritual/witchcraft causes, delaying presentation; caregiver burden is high; specialist services are concentrated in capital cities.
- **Treatment gap:** estimated >90% of people with dementia in SSA have never received a diagnosis or targeted care.
- **Etiology mix differs from HICs:** greater contribution of vascular disease, untreated hypertension, stroke sequelae, malnutrition/underweight, sensory impairment, and (in high-HIV settings) HIV-associated neurocognitive disorder (HAND); Alzheimer-type pathology is under-characterised due to near-absence of biomarker access.

### 2.3 Regional state (Sub-Saharan Africa / SADC)

- **Nigeria:** Ibadan–Indianapolis programme (Hendrie et al.) found age-standardised dementia incidence lower among Yoruba than African-Americans — a foundational transnational ageing study.
- **Kenya & Tanzania:** well-developed research nodes (e.g., Kiliﬁ Brain & Mind programme, HITONGO/IDEA work in Hai district) contributing incidence, criterion-validation, and intervention data.
- **South Africa (SADC leader):**
  - HAALSI cohort (Agincourt, rural Limpopo): dementia prevalence ~8% among older adults (de Jager et al., 2017, _Neurology_) using lay-worker cognitive screens plus clinician consensus.
  - Urban Xhosa and Cape Town studies report comparable mid-to-high single-digit to double-digit prevalence.
  - HAALSI demonstrates the feasibility of large, population-representative ageing cohorts in SADC.
- **Other SADC:** Botswana, Zambia, Malawi, Mozambique have scattered small clinic-based samples; no SADC country except South Africa has a nationally representative dementia survey.
- **Cross-cutting regional drivers:** rapid demographic ageing; extreme hypertension prevalence (>60% treated/un-treated among 60+); the long shadow of HIV (ART scale-up means growing numbers of people ageing with HIV, with HAND persisting at milder levels); economic migration of working-age adults leaving elderly-headed households; collapse/fragility of chronic-disease continuity of care.

### 2.4 National state (Zimbabwe)

- **Primary epidemiological evidence is essentially absent.** A PubMed scan (Aug 2026) identified no population-based dementia prevalence/incidence study for Zimbabwe. The closest items are:
  - Allain et al. (1996, _Cent Afr J Med_): validation attempt of an Abbreviated Mental Test in 278 Zimbabwean elders, documenting shortcomings of imported screening cutoffs in low-literacy populations.
  - HIV-neuropsychiatry work (Sebit et al.; Mielke 2005) covering HIV-associated neurocognitive complications rather than late-life dementia.
- **Modeled estimates only:** WHO Global Dementia Observatory / ADI country estimates for Zimbabwe are statistical extrapolations (tens of thousands of cases), not measured counts.
- **Enabling context:** Zimbabwe has relatively good survey infrastructure (ZIMSTAT Census 2022; DHS 2015; MICS 2019; WHO STEPS NCD survey) and an established ART programme — i.e., platforms exist onto which dementia ascertainment could be added, but has not been.

### 2.5 Synthesis of gaps this project addresses

| #   | Gap                                                               | Regional scope | This project's response                                                                                                  |
| --- | ----------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| G1  | No population-based dementia prevalence estimate for Zimbabwe     | National       | Defines target sampling frame + outcome definition (§4)                                                                  |
| G2  | No locally validated risk-prediction model; all tools are imports | Continental    | LR model restricted to community-measurable predictors                                                                   |
| G3  | Screening cutoffs biased by low literacy (Allain 1996)            | Regional       | Education-fair outcome algorithm (10/66-style composite); MMSE-style score quarantined from predictors (leakage control) |
| G4  | HIV ageing cohort entering dementia-risk window unstudied         | SADC           | HIV status included as candidate predictor                                                                               |
| G5  | Almost no African data in global risk models                      | Continental    | Calibrated synthetic dataset enables pipeline development before real collection                                         |

---

## 3. Study Design

- **Design (target study):** community-based cross-sectional survey with two-phase ascertainment (Phase 1 brief screen on full sample; Phase 2 detailed diagnostic subset), the design used successfully by 10/66 and HAALSI.
- **Target population:** community-dwelling adults ≥60 years, all 10 provinces of Zimbabwe (Bulawayo, Harare, Manicaland, Mashonaland Central/East/West, Masvingo, Matabeleland North/South, Midlands).
- **Sampling:** stratified multi-stage cluster sampling (province → urban/rural enumeration area → household → eligible respondent).
- **Exclusions:** acutely ill; inability to consent (assent-with-proxy procedures defined instead); refusal.
- **Outcome ascertainment:** composite of (a) CSI-D/10-66-style informant + participant cognitive interview, (b) functional decline measure, (c) clinician-consensus diagnosis — operationalised in `data.sql` as `dementia_status` (0/1).
- **Note for current phase:** while real data do not exist, development proceeds on a **calibrated synthetic dataset** (§5) so that the cleaning, preprocessing, modelling, and reporting pipeline is fully rehearsed before any real collection.

## 4. Variables

### 4.1 Dependent variable

| Variable          | Type       | Definition                                                                                                  | Measurement anchor                                |
| ----------------- | ---------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `dementia_status` | Binary 0/1 | Clinician-consensus dementia yes/no from Phase-2 assessment (or gold-standard simulation in synthetic data) | 10/66 / DSM-5 major neurocognitive disorder style |

### 4.2 Independent variables (candidate predictor pool)

**A. Sociodemographic**
| Variable | Type | Operationalisation | Expected direction | Evidence anchor |
|----------|------|--------------------|--------------------|-----------------|
| `age_years` | Integer 60+ | Years | ↑ risk | Universal (Lancet Commission 2024) |
| `sex` | Cat | Male/Female | ≈ / slight ↑ female (survival) | Mixed literature |
| `education_years` | Integer 0–16 | Completed formal schooling | ↓ risk per year | Lancet Commission; strongest early-life factor |
| `residence` | Cat | Urban/Rural | Context-dependent | HAALSI; service-access literature |
| `province` | Cat | 10 provinces | Exploratory | — |
| `marital_status` | Cat | Married/Widowed/Divorced-Separated/Never Married | Widowed ↑ | Social-support literature |
| `employment_status` | Cat | Retired/Subsistence Farming/Informal Trading/Formal/Unemployed | Exploratory | Cognitive-reserve literature |
| `monthly_household_income_usd` | Real | Self-reported monthly USD | ↓ risk | Poverty–dementia gradient |

**B. Vascular / metabolic**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `hypertension_dx` | Bin | ↑ | Strongest modifiable midlife factor (Lancet Commission) |
| `takes_bp_medication` | Bin | Confounder/modifier | — |
| `systolic_bp_mmhg`, `diastolic_bp_mmhg` | Int | ↑ | STEPS surveys |
| `diabetes_dx` | Bin | ↑ | Consistent globally & in SSA |
| `prior_stroke` | Bin | ↑↑ | Post-stroke dementia ~1 in 3–4 |
| `bmi_kg_m2` | Real | U-shaped (underweight ↑ late-life; obesity ↑ midlife) | SSA double-burden literature |

**C. Infectious (Zimbabwe-specific addition)**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `hiv_status` | Cat Pos/Neg/Unknown | ↑ (HAND + accelerated ageing) | Mielke 2005; ACTG A5199 incl. Zimbabwe site |

**D. Sensory**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `hearing_difficulty` | Bin | ↑ | Livingston 2024 (largest single modifiable factor in later life) |
| `vision_difficulty` | Bin | ↑ | Livingston 2024 |

**E. Mental health & neurological history**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `gds15_score` | Int 0–15 | ↑ (prodrome + risk) | Yesavage GDS-15; SSA validations |
| `head_injury_history` | Bin | ↑ | TBI–dementia dose-response |
| `family_history_dementia` | Bin | ↑ | Genetics (APOE etc.), sparse African data |

**F. Lifestyle**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `smoking_status` | Cat Never/Former/Current | ↑ | Global consistent |
| `alcohol_use` | Cat None/Moderate/Hazardous | J/U-shaped | Global |
| `physical_activity_level` | Cat Low/Moderate/High | ↓ with activity | Global + IPAQ-style STEPS item |
| `fruit_veg_servings_per_day` | Int | Weak ↓ | Diet literature |

**G. Psychosocial**
| Variable | Type | Expected direction | Anchor |
|----------|------|--------------------|--------|
| `lives_alone` | Bin | ↑ | Social isolation (Lancet Commission) |
| `social_engagement` | Ordinal Daily/Weekly/Rarely/Never | ↓ with engagement | Cognitive-reserve/social-cognitive literature |

**H. Quarantined variable (NOT a default predictor)**
| Variable | Type | Role |
|----------|------|------|
| `cognitive_screen_score` | Int 0–30 | MMSE/CSI-D-style screen used in **outcome ascertainment**. Including it as a predictor would leak the outcome → documented leakage trap; excluded by default, optionally used in a sensitivity analysis explicitly labelled as such. Rationale: imported cutoffs misclassify low-literacy respondents (Allain et al. 1996). |

## 5. Dataset Strategy — `data.sql` + `data.csv`

### 5.1 Provenance & disclaimer (important)

The dataset is **synthetic** — it contains **no real patient records**. It was generated procedurally (seeded, reproducible) to be _statistically plausible for a Zimbabwean 60+ community sample_, with marginal distributions anchored to published sources listed in §5.2. It exists to develop and stress-test the cleaning → preprocessing → modelling pipeline ahead of any real data. Any results obtained on it are illustrative only and must never be quoted as findings about Zimbabwe. The generator emits two artefacts: `data.sql` (loader plus data-dictionary and noise-manifest tables) and `data.csv` (flat mirror of the `dementia_screening` table; NULLs are empty fields).

### 5.2 Calibration anchors

| Feature                                                                                | Anchor value used                       | Source                                     |
| -------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------ |
| Sample n=12,000 unique participants, ages 60–100+, female-skewed                       | Ageing demographics                     | ZIMSTAT 2022 Population & Housing Census   |
| Residence ~30% urban; 10-province weighting                                            | Urban share & provincial distribution   | ZIMSTAT Census 2022                        |
| Overall dementia prevalence target ~8% (rising steeply with age)                       | SSA pooled prevalence ≥65 ≈ 6–10%       | Guerchet et al. 2016; HAALSI/de Jager 2017 |
| Hypertension common in 60+ (majority)                                                  | SSA elderly reviews; WHO STEPS Zimbabwe | STEPS 2014                                 |
| Diabetes ~5–10% adults                                                                 | STEPS/IDF SSA region                    | STEPS 2014                                 |
| Adult HIV prevalence ~11–12%; assumed lower (3–6%) in 60+ (flagged assumption)         | UNAIDS AIDSinfo country profile         | UNAIDS                                     |
| Smoking male-skewed (~15% men, <2% women)                                              | STEPS 2014                              | WHO                                        |
| High widowhood among elderly women                                                     | DHS/census patterns                     | ZIMSTAT/DHS                                |
| Low formal schooling in oldest cohorts (colonial-era access), rural & female penalties | DHS/MICS attainment patterns            | ZIMSTAT/DHS/MICS                           |
| Income in USD reflecting informality & low elderly household incomes                   | Macro context (post-2019 USD-isation)   | World Bank/ZIMSTAT                         |

Exact anchors are documented in the header comments of `data.sql`.

### 5.3 Engineered data-quality defects (noise manifest)

Defects are injected deliberately so the cleaning stage is a real exercise. Full machine-readable list in the `noise_manifest` table inside `data.sql`; summary:

1. Missingness — MCAR (~2–6%) on most fields; MAR on income (rural under-reporting); alcohol under-reporting gaps.
2. Survey sentinel codes — `-99` (refusal), `88` (don't know), `99` (not applicable) mixed with true NULLs.
3. Exact duplicates (~120 rows) and fuzzy duplicates (~48 rows, same ID, minor jitter).
4. Impossible values — ages >110 (e.g., 999), BMI extremes, diastolic ≥ systolic (transposition), education >25 yrs, GDS >15.
5. Categorical chaos — case variants, misspellings, trailing whitespace, abbreviations (`"Hre"`, `"Mat North"`), synonyms (`"Ex-smoker"` vs `"Former"`).
6. Date-format inconsistency — three coexisting formats in `interview_date`.
7. Protocol violations — a few records below the ≥60 inclusion threshold.
8. Logical contradictions — e.g., `hypertension_dx = No` while `takes_bp_medication = Yes`.

### 5.4 Signal engineering (so logistic regression has something to find)

`dementia_status` is generated from a latent logistic model whose coefficients mirror §4 expected effect sizes (age dominant; education, stroke, depression, hearing loss, HIV, isolation, family history, underweight all contribute), intercept tuned so overall prevalence ≈ 8%. The cognitive screen score is then derived partly from the same latent propensity — realistic and dangerous if leaked into the model, exactly as in real practice.

## 6. Analysis Plan

### 6.1 Pipeline stages

1. **Ingest** `data.sql` (any engine; tested on SQLite).
2. **Cleaning** driven by `noise_manifest`: deduplicate on `participant_id` (document conflict-resolution rule), repair/reject impossible values (explicit rejection log, never silent overwrite), harmonise categories via controlled vocabularies, parse dates, convert sentinels to NA, apply inclusion rule age ≥60.
3. **Preprocessing**: train/test split first (stratified 80/20, seed fixed); median/mode imputation fitted on train only; one-hot encoding (drop-first); optional standardisation for L2 paths; missing-indicator flags where MAR suspected.
4. **Modelling**: univariable screening → multivariable LR (report both crude and adjusted ORs) → regularised variants (L1/L2 via CV) → class-imbalance handling comparison (class weights vs SMOTE on train folds only) → probability calibration assessment.
5. **Validation**: repeated stratified k-fold (k=5) CV; final hold-out evaluation once.
6. **Metrics**: AUC-ROC (primary), sensitivity/specificity grid with Youden and clinically motivated points, PPV/NPV at plausible prevalence, F1, Brier score, calibration curve/intercept-slope.
7. **Interpretability**: odds ratios with CIs, forest plot, standardised coefficients, nomogram-style score card for CHW deployment.
8. **Subgroup/fairness**: performance by sex and urban/rural; calibration drift check.
9. **Sensitivity analyses**: (a) complete-case vs imputed; (b) with/without `cognitive_screen_score` to demonstrate leakage inflation; (c) exclusion of contradictory/BP-transposed records; (d) age-restricted ≥65 variant matching SSA prevalence literature.

### 6.2 Sample size logic

Events-per-variable rule (EPV ≥ 10): with ~20 df of predictors and ~8% prevalence, need ≥250 events → ≥~3,100 screened for the eventual real study. The synthetic sample is n=12,000 unique participants (~930 events at ~7.8% prevalence), comfortably above EPV ≥ 10 for the ~20-df predictor pool; the earlier 1,500-row draft was deliberately EPV-deficient (~6) — subsample from the full table to reproduce that small-event teaching scenario. Regularisation and penalisation exercises remain in scope.

### 6.3 Tooling

Python ≥3.10; pandas, scikit-learn, statsmodels, matplotlib; SQLite via stdlib `sqlite3`. No GPU required — LR on tabular data is CPU-trivial.

## 7. Ethics & Governance (for future real-data phase)

- Approval: Medical Research Council of Zimbabwe (MRCZ) + institutional review boards; provincial/district health authorities.
- Consent: capacity-assessed consent; proxy assent + participant assent where capacity impaired; separate imaging/biobanking consent if ever added.
- Community engagement before fieldwork; stigma-sensitive framing (dementia ≠ madness/witchcraft).
- Data protection: Zimbabwe Cyber and Data Protection Act [Chapter 12:07], 2021 compliance; pseudonymised IDs (as in this dataset — no names/locations beyond province), encrypted storage, local custodianship.
- Benefit-sharing: feedback of aggregate findings; referral pathway for screen-positive individuals.

## 8. Limitations & Risks

| Risk                                                              | Mitigation                                                                           |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Cross-sectional design cannot establish temporality               | Frame as prediction/screening, not causal inference; longitudinal follow-on proposed |
| Outcome ascertainment error (no gold standard without biomarkers) | Two-phase design; criterion-comparison sub-study                                     |
| Literacy bias in cognitive tests                                  | Education-fair battery; document score-by-literacy interaction                       |
| Class imbalance (~8%)                                             | Weights/resampling; report PR-AUC alongside ROC-AUC                                  |
| Overfitting on small event count                                  | Regularisation; EPV discipline; honest internal validation only                      |
| Synthetic-to-real transfer illusion                               | All synthetic results labelled illustrative; pipeline re-run mandatory on real data  |

## 9. Roadmap

| Phase      | Milestone                                      | Output                                                        |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------- |
| 0 (done)   | Planning + synthetic dataset                   | `plan.md`, `data.sql`, `data.csv`, generator script           |
| 1          | Cleaning module                                | `01_cleaning.py` + clean table + rejection log                |
| 2          | EDA report                                     | distributions, missingness map, noise-manifest reconciliation |
| 3          | Baseline LR + CV                               | metrics table, OR forest plot                                 |
| 4          | Imbalance & regularisation experiments         | comparison report                                             |
| 5          | Leakage demo & sensitivity analyses            | teaching note                                                 |
| 6          | Final hold-out evaluation + scorecard artefact | model card + limitations statement                            |
| 7 (future) | Grant/ethics package for real Zimbabwean data  | protocol v2                                                   |

## 10. Key References (selected)

1. WHO. _Dementia_ fact sheet. Geneva: World Health Organization (2025 update; 57M in 2021; ~10M new cases/yr; >60% in LMICs; US$1.3tn cost 2019).
2. Livingston G, et al. _Dementia prevention, intervention, and care: 2024 report of the Lancet standing Commission._ Lancet (2024). [14 modifiable factors; ~45% potentially preventable]
3. Prince M, et al. _The 10/66 Dementia Research Group's fully operationalised CRITERIA…_ Br J Psychiatry (2007); and World Alzheimer Report series, ADI.
4. Guerchet M, et al. Dementia in Sub-Saharan Africa — prevalence reviews (approx. 2016).
5. de Jager CA, et al. Dementia prevalence in rural South Africa (HAALSI/Agincourt). Neurology (2017) — approx. 8%.
6. Hendrie HC, et al. Ibadan–Indianapolis study (JAMA 1995; Neurology 2001) — Yoruba vs African-American incidence.
7. Longdon AR, Paddick SM, et al. Rates of dementia and criterion comparison in SSA (Tanzania IDEA/HITONGO line of work, approx. 2013–2020).
8. Allain TJ, Wilson AO, Gomo ZA, Adamchak DJ, Matenga JA. Abbreviated Mental Test in the elderly: shortcomings of an adapted AMT in Zimbabwe. Cent Afr J Med (1996).
9. Mielke J. Neurological complications of HIV infection in Zimbabwe. J Neurovirol (2005).
10. ZIMSTAT. 2022 Population and Housing Census; DHS 2015; MICS 2019.
11. WHO. Zimbabwe STEPwise survey for noncommunicable disease risk factors (2014).
12. UNAIDS. AIDSinfo country profile: Zimbabwe (adult HIV prevalence approx. 11–12%).
13. Republic of Zimbabwe. Cyber and Data Protection Act [Chapter 12:07] (2021).

_(Items 4, 7, and some figures marked "approx." could not be re-verified online at planning time and should be citation-checked against the originals before publication.)_
