# modelx.ipynb Assessment Report
**Project:** ml_dementia (Zimbabwe Dementia Risk Prediction)  
**Notebook:** modelx.ipynb — Logistic Regression Pipeline  
**Date:** 2026-08-29  
**Data:** Synthetic `data.csv` (12,168 rows × 31 cols) mirroring `dementia_screening` table  
**Target:** `dementia_status` (~8% prevalence)

---

## 1. Data Analysis, Cleaning & Visuals — Deep Dive

### 1.1 Ingest & First Look (§2–§3)

| Aspect | Finding |
|--------|---------|
| **Shape** | 12,168 rows × 31 columns |
| **Dtypes** | 13 float64, 6 int64, 12 object/str |
| **Key Issue** | `keep_default_na=False` preserves `"None"` (alcohol_use) as label, not NaN — correct choice |
| **Immediate Red Flags** | Age max 999, BMI 102.5, education 51, negative fruit/veg servings — all engineered defects per noise manifest |

**Visual Audit (Raw State):**
- **Missingness (§3.2):** `monthly_household_income_usd` 14.3% missing (MAR: rural under-reporting), `diastolic_bp_mmhg` 9.8%, `alcohol_use` 6.4%
- **Distributions (§3.3):** Clear sentinel pile-ups at -99, 88, 99; age spike at 999; BMI spike at 102.5; education spike at 51
- **Categorical Chaos (§3.4):** 38 raw province levels → 10 canonical; marital_status 18 → 4; smoking 15 → 3; alcohol_use 15 → 3
- **Crude Associations (§3.5):** Expected gradients — dementia prevalence rises with age band, falls with education band, higher with hearing difficulty
- **Correlation (§3.6):** `cognitive_screen_score` ↔ `dementia_status` = -0.28 (leakage risk → quarantined per plan §4.2H)

### 1.2 Cleaning Pipeline (§4) — Manifest-Driven, Audit-First

| Step | Action | Rows Affected | Rationale |
|------|--------|---------------|-----------|
| **4.1 Whitespace trim + dedup** | Exact dups (120) dropped; fuzzy dups on `participant_id` (48) → keep first | 168 removed | Conflict rule documented |
| **4.2 Date parsing** | Multi-format parse (ISO, dd/mm/yyyy, MMM dd, yyyy) | 0 unparseable | Dates retained for audit, not modelling |
| **4.3 Sentinels → NaN** | -99/88/99 on income, education, GDS-15, fruit/veg | 291+242+261+203 cells | Only manifest-specified columns touched |
| **4.4 Range checks** | OOB → NaN (BMI 40, edu 32, GDS 24, servings 16, SBP 6, DBP 15) | Cells flagged | Physiological bounds per manifest |
| **4.4b BP transposition** | Diastolic ≥ systolic → swap if plausible | 61 found, 61 repaired | Silent swap would be unsafe; explicit count |
| **4.4c Age rejection** | Age >110 or missing | 56 rows rejected | Eligibility unverifiable |
| **4.5 Categorical harmonisation** | Case-fold + synonym maps → 10 provinces, 3 smoking, 3 alcohol, etc. | 21 province labels unmapped → NaN | Controlled vocabularies per plan §4.2 |
| **4.6 Logical contradiction** | `takes_bp_med=1` & `hypertension_dx=0` → med set to NaN | 79 cells | Diagnosis field more reliable |
| **4.7 Eligibility** | Age < 60 → reject | 20 rows | Protocol violation |

**Final Clean Dataset:** 11,924 rows (244 removed = 76 logged + 168 dups), prevalence 8.0% (959 events)  
**EPV Check:** 959 events ≥ 10 × ~35 model df → satisfied

**Post-Cleaning Visuals (§4.8):**
- Missingness shifts: income 16.7% (sentinels converted), GDS-15 5.6% (was 3.3%)
- Age distribution clean (60–100+), province/sex/smoking harmonised
- Exports: `outputs/clean_data.csv`, `outputs/rejection_log.csv`

---

### 1.3 Suggested Additional Analytics (Gaps in Current EDA)

| Missing Analysis | Why It Matters | Suggested Implementation |
|------------------|----------------|--------------------------|
| **Missingness pattern heatmap** | MAR diagnosis (income rural) needs visual proof of association with `residence` | `missingno.matrix(clean)` or seaborn heatmap of `clean.isna().corr()` |
| **Pairwise numeric plots by outcome** | Detect non-linearities, interactions (e.g., age × education) | `sns.pairplot(clean[num_cols + [TARGET]], hue=TARGET, diag_kind="kde")` |
| **Target leakage check on ALL predictors** | Cognitive screen quarantined, but others? (e.g., `gds15_score` may correlate with ascertainment) | Compute `mutual_info_classif` or `pointbiserialr` for each predictor vs outcome |
| **Stratified missingness by outcome** | MNAR risk: are dementia cases more/less likely to have missing income? | `clean.groupby(TARGET).apply(lambda g: g.isna().mean())` |
| **Outlier influence diagnostics** | Cook's distance / leverage on top predictors (age, prior_stroke) | Statsmodels `get_influence()` after unpenalised fit |
| **Temporal drift check** | Interview dates span 2024-01 to 2025-03; any prevalence shift? | `clean.groupby(clean.interview_date.dt.to_period('M'))[TARGET].mean().plot()` |
| **Interaction screening** | Age × hearing_difficulty, education × social_engagement are clinically plausible | Fit LR with pre-specified interactions, compare AIC |
| **Feature importance permutation** | Model-agnostic check on LR coefficients | `sklearn.inspection.permutation_importance` on hold-out |

---

## 2. Model Evaluation — Detailed Findings

### 2.1 Cross-Validation (Train, 5-Fold Stratified) — §6.1

| Metric | Mean ± SD | Plan Target | Interpretation |
|--------|-----------|-------------|----------------|
| **ROC-AUC** | 0.785 ± 0.011 | ≥ 0.75 (H2) | ✅ Passes — good discrimination, low fold variance |
| **PR-AUC** | 0.342 ± 0.033 | — | Moderate; reflects 8% prevalence (baseline = 0.08) |
| **Brier Score** | 0.063 ± 0.002 | — | Well-calibrated probabilities (ideal ≤ prevalence×(1-prevalence) = 0.074) |

**Fold-level stability:** ROC-AUC range [0.766, 0.799] — no fold collapse, consistent signal.

### 2.2 Class Imbalance Handling — §6.2

| Model | CV ROC-AUC | CV Recall | CV Precision |
|-------|------------|-----------|--------------|
| LR (default) | 0.785 | — | — |
| LR (balanced weights) | 0.782 | 0.658 | 0.203 |

**Decision:** Kept default weights. Rationale:
- Balanced weights inflate predicted probabilities → poor Brier/calibration
- Screening operating point set by **explicit threshold** (sensitivity ≥ 0.80), not by reweighting
- This is correct for a screening tool where threshold tuning is clinically driven

### 2.3 Hold-Out Evaluation (n=2,385, 8.1% events) — §6.3

| Operating Point | Threshold | Sensitivity | Specificity | PPV | NPV |
|-----------------|-----------|-------------|-------------|-----|-----|
| **Youden's J** | 0.095 | 0.620 | 0.810 | 0.222 | 0.961 |
| **Screening (sens ≥ 0.80)** | 0.040 | **0.802** | 0.511 | 0.126 | 0.967 |

**Key Observations:**
- **Hold-out ROC-AUC = 0.771** (vs CV 0.785) — mild optimism correction, expected
- **PR-AUC = 0.337** — 4.2× baseline prevalence (0.08), useful for imbalanced setting
- **Brier = 0.063** — excellent calibration (well below null 0.074)
- **Screening threshold (0.040) is very low** — captures 80% of cases but flags ~49% of non-cases (low specificity). This is typical for community screening; PPV 12.6% means ~8 referrals per true case.
- **NPV 96.7%** at screening threshold — strong rule-out value

**Classification Report (Youden J):**
```
              precision    recall  f1-score   support
 No dementia       0.96      0.81      0.88      2193
    Dementia       0.22      0.62      0.33       192
    accuracy                           0.79      2385
   macro avg       0.59      0.72      0.60      2385
weighted avg       0.90      0.79      0.83      2385
```

### 2.4 Calibration — §6.4

- **Calibration slope = 1.04** (ideal 1.0) — slight underfitting, but very close
- **Calibration intercept = 0.02** (ideal 0.0) — near-perfect
- **Brier = 0.063** — consistent with good calibration
- **Visual:** 10-bin quantile curve tracks diagonal closely; no systematic over/under-confidence

**Implication:** Raw probabilities are trustworthy for risk communication; no Platt/isotonic recalibration needed.

### 2.5 Interpretation — Odds Ratios (Unpenalised, statsmodels) — §7

| Predictor | OR (95% CI) | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| **prior_stroke** | 2.26 (1.70, 2.99) | <0.001 | Strongest predictor; 2.3× odds per history |
| **age_years (per 1 SD)** | 2.06 (1.91, 2.22) | <0.001 | ~2× odds per ~8.5 yr increase (SD) |
| **family_history_dementia** | 1.94 (1.50, 2.52) | <0.001 | Nearly 2× odds with positive family history |
| **hearing_difficulty** | 1.61 (1.30, 1.99) | <0.001 | Modifiable risk factor; 61% higher odds |
| **hypertension_dx** | 1.50 (1.17, 1.93) | 0.002 | 50% higher odds |
| **diabetes_dx** | 1.43 (1.11, 1.84) | 0.005 | 43% higher odds |
| **lives_alone** | 1.42 (1.16, 1.73) | 0.001 | Social isolation signal |
| **vision_difficulty** | 1.37 (1.13, 1.67) | 0.002 | Sensory deficit association |
| **gds15_score (per 1 SD)** | 1.13 (1.04, 1.22) | 0.005 | Depressive symptoms — modest effect |
| **HIV Positive** | 1.35 (0.91, 2.01) | 0.132 | Directional but CI crosses 1 |
| **SBP/DBP (per 1 SD)** | ~1.03–1.04 | NS | No independent BP effect after meds/diagnosis |

**Model Fit:** AIC = 4,427 | McFadden pseudo-R² = 0.189 (moderate explanatory power)

**Forest Plot:** Visual confirms clean separation of significant predictors (CI excludes 1) from null.

### 2.6 Subgroup Performance — §8

| Subgroup | n | Events | ROC-AUC | Mean Pred | Observed |
|----------|---|--------|---------|-----------|----------|
| Sex: Male | 1,070 | 83 | **0.791** | 0.075 | 0.078 |
| Sex: Female | 1,315 | 109 | **0.752** | 0.088 | 0.083 |
| Residence: Rural | 1,579 | 134 | **0.763** | 0.086 | 0.085 |
| Residence: Urban | 806 | 58 | **0.788** | 0.076 | 0.072 |

**Findings:**
- **Discrimination drift:** Male AUC 0.791 vs Female 0.752 (Δ=0.039); Urban 0.788 vs Rural 0.763 (Δ=0.025)
- **Calibration:** Mean predicted ≈ observed in all strata — well-calibrated across subgroups
- **Caveat:** Wide CIs on smaller strata (Urban events=58) — differences indicative, not definitive
- **H3 Verdict:** Stable AUC *approximately* holds; calibration stable; sex disparity warrants monitoring

### 2.7 Leakage Sensitivity Analysis — §9

| Model | Hold-out ROC-AUC | Δ |
|-------|------------------|---|
| Quarantined (correct) | 0.771 | — |
| Leaked `cognitive_screen_score` | 0.771 | +0.000 |

**Key Insight:** In *this* synthetic data, the screen adds nothing once age/education are in the model (OR=0.96). However, the **conceptual trap remains**: the screen is used in outcome ascertainment (plan §4.2H), so any conditional association is circular. With richer cognitive items, inflation would be dramatic. Quarantining is correct practice.

---

## 3. Fine-Tuning Opportunities

### 3.1 Immediate (Low Effort, High Impact)

| Opportunity | Current State | Proposed Change | Expected Gain |
|-------------|---------------|-----------------|---------------|
| **Threshold optimisation for deployment** | Fixed Youden/screening points | Add cost-sensitive threshold: minimise `c_fn × FN + c_fp × FP` with local referral costs | Aligns operating point to health-system economics |
| **Penalty strength (C) tuning** | Default `C=1.0` (L2) | Nested CV for `C ∈ [0.01, 0.1, 1, 10, 100]` on train folds | May improve AUC/Brier slightly; prevents overfit |
| **Firth/bias-reduced LR** | Standard MLE (statsmodels) | `statsmodels.Logit(..., method='bfgs').fit_regularized(alpha=0, method='firth')` | Better small-sample ORs, especially for rare predictors (prior_stroke n≈100) |
| **Interaction terms (pre-specified)** | None | Add `age × hearing_difficulty`, `education × social_engagement`, `HIV × residence` | Clinical plausibility; check AIC/LRT |
| **Income MAR handling enhancement** | Median impute + missing flag | Try MICE (IterativeImputer) or target-encoding income by province/residence | May recover signal from 16.7% missing income |

### 3.2 Moderate Effort (Roadmap Phase 4–5)

| Opportunity | Description |
|-------------|-------------|
| **SMOTE / ADASYN on train folds only** | Compare balanced-weight vs oversampling vs undersampling; evaluate on hold-out with *same* threshold |
| **Complete-case vs imputed sensitivity** | Fit on rows with zero missingness (n≈8,500) — compare coefficients, AUC |
| **Age-restricted variant (≥65)** | Re-run pipeline on age ≥ 65 subset; check if ORs shift (competing risk) |
| **Elastic Net (L1+L2)** | `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)` for automatic feature selection |
| **Calibration refinement** | Fit Platt scaling (logistic on logits) or isotonic regression on hold-out; report ECE |
| **Nomogram / CHW scorecard** | Convert LR coefficients to integer points (Sullivan method) for paper-based screening |

### 3.3 Advanced / Research-Grade

| Opportunity | Description |
|-------------|-------------|
| **Competing risk model** | Dementia vs death (if mortality data available) — Fine-Gray subdistribution hazards |
| **Time-to-event / survival** | If follow-up dates exist: Cox PH with same covariates |
| **External validation** | Apply frozen model to independent Zimbabwe cohort (when real data arrives) |
| **Causal mediation** | Decompose hearing_difficulty → social_engagement → dementia pathway |
| **Fairness audit** | Equalized odds / calibration by sex, residence, HIV status; mitigate if disparities > threshold |
| **Bayesian LR with informative priors** | Incorporate published meta-analysis ORs as priors (e.g., age, stroke) for small-sample stability |

---

## 4. Summary Scorecard

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| **Data Cleaning Rigour** | 5 | Manifest-driven, explicit rejection log, no silent repairs |
| **Leakage Prevention** | 5 | Quarantined cognitive screen, split-first, train-only fit |
| **EDA Completeness** | 4 | Strong; missing pairplots, missingness patterns, temporal drift |
| **Model Validation** | 5 | Stratified CV + hold-out, dual thresholds, calibration, subgroups |
| **Interpretability** | 5 | Unpenalised ORs with CIs, forest plot, standardised coefficients |
| **Class Imbalance Handling** | 4 | Correct threshold-based approach; could add cost-sensitive tuning |
| **Reproducibility** | 5 | Fixed seed, versioned exports, single notebook pipeline |
| **Documentation** | 5 | Inline plan references, rationale for every decision |

**Overall: Production-grade prototype pipeline for synthetic data rehearsal.**

---

## 5. Recommended Next Actions (Priority Order)

1. **Add threshold cost-optimisation** — define local `c_fn`/`c_fp` with clinical stakeholders
2. **Run nested CV for `C` (penalty strength)** — 5×5 nested, report mean hold-out AUC
3. **Fit Firth LR** — compare ORs/CIs for rare predictors (stroke, HIV+)
4. **Pre-specified interaction screen** — 3 clinical interactions, LRT vs main effects
5. **Complete-case sensitivity** — document coefficient stability
6. **Generate model card** — per Mitchell et al. 2019 template (intended use, limitations, ethical considerations)
7. **Prepare for external validation** — freeze `preprocessor`, `final_model`, `threshold_screen` as pickle/ONNX

---

## Appendix: Key Files & Artefacts

| File | Purpose |
|------|---------|
| `data.csv` | Raw synthetic data (12,168 × 31) |
| `outputs/clean_data.csv` | Cleaned analysis dataset (11,924 × 31) |
| `outputs/rejection_log.csv` | Row-level audit trail (76 rows) |
| `outputs/odds_ratios.csv` | Unpenalised LR ORs with 95% CIs |
| `plan.md` | Analysis plan (source of truth for all § references) |
| `modelx.ipynb` | This notebook — full pipeline |

---