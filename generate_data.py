"""
generate_data.py
================
Builds `data.sql` + `data.csv`: a synthetic, Zimbabwe-calibrated dementia screening dataset
with deliberately engineered data-quality defects for the cleaning/preprocessing
stages of the ml_dementia project.

- Stdlib only (math, random, csv, datetime). No external dependencies.
- Deterministic: random.seed(42) -> regenerate any time, byte-identical output.
- The dataset is 100% SYNTHETIC. No real patient records. Marginal distributions
  are calibrated to published sources listed in plan.md section 5.2 and in the
  header comments of the emitted SQL file.

Usage:  python generate_data.py
Output: data.sql and data.csv in this script's directory.
"""

import math
import random
import os
import csv
from collections import Counter
from datetime import date, timedelta

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "data.sql")
CSV_PATH = os.path.join(HERE, "data.csv")

N = 12000                     # unique participants
TARGET_PREVALENCE = 0.078     # ~8%, SSA-calibrated

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bern(p):
    return random.random() < p


def wchoice(pairs):
    r = random.random()
    acc = 0.0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]


def poisson(lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1


def pick_province():
    r = random.random()
    acc = 0.0
    for name, w, kind in PROVINCES:
        acc += w
        if r <= acc:
            return name, kind
    return PROVINCES[-1][0], PROVINCES[-1][2]


# ----------------------------------------------------------------------------
# Population structure (calibration anchors: ZIMSTAT Census 2022)
# ----------------------------------------------------------------------------

PROVINCES = [
    ("Harare", 0.195, "city"),
    ("Bulawayo", 0.065, "city"),
    ("Manicaland", 0.125, "rural"),
    ("Mashonaland Central", 0.072, "rural"),
    ("Mashonaland East", 0.098, "mixed"),
    ("Mashonaland West", 0.105, "mixed"),
    ("Masvingo", 0.112, "rural"),
    ("Matabeleland North", 0.068, "rural"),
    ("Matabeleland South", 0.062, "rural"),
    ("Midlands", 0.098, "mixed"),
]

URBAN_P = {"city": 0.88, "mixed": 0.22, "rural": 0.09}

# ----------------------------------------------------------------------------
# Category corruption vocabularies (noise)
# ----------------------------------------------------------------------------

SEX_V = {
    "M": ["M", "m", "MALE", "male", "Male ", " Male", "M."],
    "F": ["F", "f", "FEMALE", "female", "Female ", " female"],
}
RES_V = {
    "Urban": ["Urban", "URBAN", "urban ", "Urbn"],
    "Rural": ["Rural", "RURAL", "rural ", "Rual"],
}
PROV_V = {
    "Harare": ["HARARE", " harare", "Hre", "Harare Province"],
    "Bulawayo": ["BULAWAYO", "Bulawayo ", "bulawayo"],
    "Manicaland": ["manicaland", "Manicaland ", "MANICALAND"],
    "Mashonaland Central": ["mashonaland central", "Mashonaland  Central", "Mash Central"],
    "Mashonaland East": ["Mashonaland east", "MASHONALAND EAST", "Mash East"],
    "Mashonaland West": ["mashonaland west", "Mash West", "Mashonaland  West"],
    "Masvingo": ["masvingo", "MASVINGO ", "Masvingo "],
    "Matabeleland North": ["Mat North", "matabelaland north", "Matabeleland  North", "MAT NORTH"],
    "Matabeleland South": ["Mat South", "matabeleland south ", "MATABELALAND SOUTH"],
    "Midlands": ["midlands", "MIDLANDS", "Midlands "],
}
MARITAL_V = {
    "Married": ["married", "MARRIED", "maried", "Married "],
    "Widowed": ["widowed", "WIDOWED", "widwoed", "Widow"],
    "Divorced/Separated": ["Divorced", "divorced/seperated", "Divorced/Separated ", "divorced / separated"],
    "Never Married": ["Single", "single", "NEVER MARRIED", "never-married"],
}
EMPLOY_V = {
    "Retired": ["retired", "RETIRED", "Retired "],
    "Subsistence Farming": ["Subsistence farming", "subsistence farmer", "Farming", "SUBSISTENCE FARMING"],
    "Informal Trading": ["informal trade", "Informal trade", "VENDOR", "vendor", "Informal Trading "],
    "Formal Employment": ["Formal", "formal sector", "FORMAL EMPLOYMENT"],
    "Unemployed": ["unemployed", "UNEMPLOYED ", "Not employed"],
}
SMOKE_V = {
    "Never": ["never", "NEVER", "never smoked", "None"],
    "Former": ["former", "Ex-smoker", "ex smoker", "FORMER"],
    "Current": ["current", "CURRENT SMOKER", "current smoker", "Yes"],
}
ALCOHOL_V = {
    "None": ["none", "NONE ", "Non", "no"],
    "Moderate": ["moderate", "MODERATE", "Light", "light"],
    "Hazardous": ["hazardous", "Heavy", "HAZARDOUS ", "heavy drinker"],
}
SOCIAL_V = {
    "Daily": ["daily", "DAILY"],
    "Weekly": ["weekly", "WEEKLY"],
    "Rarely": ["rarely", "RARELY"],
    "Never": ["never", "Never "],
}
HIV_V = {
    "Positive": ["positive", "POS", "Positive "],
    "Negative": ["negative", "neg", "NEGATIVE"],
    "Unknown": ["unknown", "UNKNOWN"],
}

DATE_FMTS = ["iso", "dmy", "long"]  # weighted below

# ----------------------------------------------------------------------------
# Person generation (clean ground truth first; noise applied afterwards)
# ----------------------------------------------------------------------------


def make_person(i):
    row = {}
    prov, kind = pick_province()
    residence = "Urban" if bern(URBAN_P[kind]) else "Rural"
    sex = "F" if bern(0.56) else "M"
    age = 60 + min(42, int(random.expovariate(1 / 8.5)))
    birth_year = 2026 - age

    # Education: colonial-era cohort gradients + rural + female penalties
    if birth_year <= 1946:
        base_edu = 2.2
    elif birth_year <= 1956:
        base_edu = 4.8
    elif birth_year <= 1966:
        base_edu = 7.6
    else:
        base_edu = 8.5
    if residence == "Rural":
        base_edu -= 1.6
    if sex == "F":
        base_edu -= 1.4
    edu = int(clamp(round(random.gauss(base_edu, 2.8)), 0, 16))

    # Employment
    if residence == "Rural":
        employment = wchoice([
            ("Retired", 0.45), ("Subsistence Farming", 0.40),
            ("Informal Trading", 0.07), ("Unemployed", 0.05),
            ("Formal Employment", 0.03)])
    else:
        employment = wchoice([
            ("Retired", 0.62), ("Informal Trading", 0.18),
            ("Unemployed", 0.10), ("Formal Employment", 0.08),
            ("Subsistence Farming", 0.02)])

    # Income (USD/month, lognormal, informal economy)
    mu_inc = math.log(55) if residence == "Rural" else math.log(75)
    if employment == "Formal Employment":
        mu_inc = math.log(180)
    income = int(clamp(round(math.exp(random.gauss(mu_inc, 0.95)) / 5) * 5, 5, 800))

    # Marital status & living arrangement
    widow_p = clamp(0.10 + 0.008 * (age - 60) + (0.30 if sex == "F" else 0.04), 0.02, 0.85)
    marital = wchoice([
        ("Married", max(0.02, 1 - widow_p - 0.09)),
        ("Widowed", widow_p),
        ("Divorced/Separated", 0.06),
        ("Never Married", 0.03)])
    lives_alone = int(bern(0.45 if marital == "Widowed" else 0.10))

    social = wchoice([
        ("Daily", max(0.05, 0.40 - (0.10 if lives_alone else 0))),
        ("Weekly", 0.35),
        ("Rarely", 0.20 + (0.05 if lives_alone else 0)),
        ("Never", 0.05 + (0.10 if lives_alone else 0))])

    # HIV (assumption: lower than adult average among 60+; flagged in plan.md)
    hiv_p = clamp(0.045 - 0.0005 * (age - 60), 0.01, 0.10)
    hiv_status = wchoice([("Positive", hiv_p), ("Negative", 1 - hiv_p)])

    # Lifestyle
    if sex == "M":
        smoking = wchoice([("Never", 0.74), ("Former", 0.11), ("Current", 0.15)])
        alcohol = wchoice([("None", 0.55), ("Moderate", 0.33), ("Hazardous", 0.12)])
    else:
        smoking = wchoice([("Never", 0.975), ("Former", 0.015), ("Current", 0.01)])
        alcohol = wchoice([("None", 0.76), ("Moderate", 0.22), ("Hazardous", 0.02)])

    if residence == "Rural":
        activity = wchoice([("High", 0.58), ("Moderate", 0.32), ("Low", 0.10)])
    else:
        activity = wchoice([("High", 0.24), ("Moderate", 0.46), ("Low", 0.30)])

    fruit_veg = int(clamp(round(random.gauss(3.0, 1.4)), 0, 8))

    # Anthropometrics (double burden: underweight elderly vs obesity)
    bmi_mu = 21.5
    bmi_mu += 1.8 if activity == "Low" else (-1.0 if activity == "High" else 0)
    bmi_mu += 1.5 if income > 200 else (0.8 if income > 100 else 0)
    bmi = clamp(random.gauss(bmi_mu, 3.6), 13.5, 45)

    # Vascular/metabolic
    hyp_p = clamp(0.45 + 0.008 * (age - 60) + (0.05 if bmi >= 30 else 0), 0, 0.90)
    hypertension = int(bern(hyp_p))
    takes_bp_med = int(bern(0.45)) if hypertension == 1 else 0
    if hypertension == 1:
        sys_bp = random.gauss(142 if takes_bp_med else 150, 16)
        dia_bp = random.gauss(89 if takes_bp_med else 93, 10)
    else:
        sys_bp = random.gauss(124, 14)
        dia_bp = random.gauss(78, 9)
    sys_bp = int(clamp(round(sys_bp), 80, 260))
    dia_bp = int(clamp(round(dia_bp), 45, 140))

    diabetes = int(bern(clamp(0.055 + 0.0025 * (age - 60) + (0.10 if bmi >= 30 else 0), 0, 0.35)))
    stroke = int(bern(clamp(0.02 + 0.0012 * (age - 60) + (0.035 if hypertension else 0), 0, 0.30)))

    head_injury = int(bern(0.05))
    family_hist = int(bern(0.08))

    # Depression (GDS-15 style score 0-15)
    gds_lam = 1.2 + (1.1 if lives_alone else 0) + (0.9 if marital == "Widowed" else 0) + 0.02 * (age - 60)
    gds = min(15, poisson(gds_lam))

    hearing = int(bern(clamp(0.04 + 0.008 * (age - 60) + (0.03 if sex == "M" else 0), 0, 0.90)))
    vision = int(bern(clamp(0.10 + 0.007 * (age - 60), 0, 0.85)))

    row.update({
        "participant_id": "ZW-%05d" % i,
        "_eta_inputs": None,
        "province": prov,
        "residence": residence,
        "sex": sex,
        "age_years": age,
        "marital_status": marital,
        "education_years": edu,
        "employment_status": employment,
        "monthly_household_income_usd": float(income),
        "lives_alone": lives_alone,
        "social_engagement": social,
        "hiv_status": hiv_status,
        "hypertension_dx": hypertension,
        "takes_bp_medication": takes_bp_med,
        "diabetes_dx": diabetes,
        "prior_stroke": stroke,
        "head_injury_history": head_injury,
        "family_history_dementia": family_hist,
        "systolic_bp_mmhg": sys_bp,
        "diastolic_bp_mmhg": dia_bp,
        "bmi_kg_m2": round(bmi, 1),
        "gds15_score": gds,
        "hearing_difficulty": hearing,
        "vision_difficulty": vision,
        "smoking_status": smoking,
        "alcohol_use": alcohol,
        "physical_activity_level": activity,
        "fruit_veg_servings_per_day": fruit_veg,
    })

    # Interview date within fieldwork window
    start = date(2024, 1, 8)
    d = start + timedelta(days=random.randrange((date(2025, 3, 31) - start).days + 1))
    row["_date"] = d

    # Latent linear predictor (effect sizes per plan.md section 6 / literature)
    eta = (
        0.085 * (age - 60)
        + (0.15 if sex == "F" else 0.0)
        + 0.055 * max(0, 5 - edu)
        + 0.35 * hypertension
        + 0.30 * diabetes
        + 0.80 * stroke
        + 0.09 * gds
        + (0.50 if hiv_status == "Positive" else 0.0)
        + 0.40 * hearing
        + 0.30 * vision
        + (0.35 if activity == "Low" else 0.0)
        + 0.30 * lives_alone
        + 0.50 * family_hist
        + (0.40 if bmi < 18.5 else 0.0)
        + 0.15 * head_injury
    )
    return row, eta


# ----------------------------------------------------------------------------
# Build clean sample and calibrate intercept to target prevalence
# ----------------------------------------------------------------------------

rows, etas = [], []
for i in range(1, N + 1):
    row, eta = make_person(i)
    rows.append(row)
    etas.append(eta)


def mean_p(c):
    return sum(sigmoid(e + c) for e in etas) / len(etas)


lo, hi = -8.0, 2.0
for _ in range(60):
    mid = (lo + hi) / 2.0
    if mean_p(mid) < TARGET_PREVALENCE:
        lo = mid
    else:
        hi = mid
INTERCEPT = (lo + hi) / 2.0

for row, eta in zip(rows, etas):
    p_latent = sigmoid(eta + INTERCEPT)
    row["dementia_status"] = int(bern(p_latent))
    # Cognitive screen (MMSE/CSI-D style): education-biased, outcome-correlated.
    mu_scr = 28.5 - 0.09 * (row["age_years"] - 60) \
             - 0.7 * max(0, 4 - row["education_years"]) - 8.8 * p_latent
    row["cognitive_screen_score"] = int(clamp(round(random.gauss(mu_scr, 1.8)), 0, 30))

achieved_prev = sum(r["dementia_status"] for r in rows) / N

# ----------------------------------------------------------------------------
# Noise injection
# ----------------------------------------------------------------------------

cnt = Counter()


def fmt_date(d, fmt):
    if fmt == "iso":
        return d.isoformat()
    if fmt == "dmy":
        return "%02d/%02d/%04d" % (d.day, d.month, d.year)
    return "%s %d, %d" % (d.strftime("%b"), d.day, d.year)


def corrupt(value_map, key, prob):
    variants = value_map.get(key)
    if variants and bern(prob):
        cnt["category_variants"] += 1
        return random.choice(variants)
    return key


TEXT_PAD_PROB = 0.04

for r_idx, row in enumerate(rows):
    # Date format inconsistency (three coexisting formats)
    fmt = wchoice([("iso", 0.60), ("dmy", 0.25), ("long", 0.15)])
    row["interview_date"] = fmt_date(row.pop("_date"), fmt)

    # Categorical chaos
    row["sex"] = corrupt(SEX_V, row["sex"], 0.07)
    row["residence"] = corrupt(RES_V, row["residence"], 0.05)
    row["province"] = corrupt(PROV_V, row["province"], 0.05)
    row["marital_status"] = corrupt(MARITAL_V, row["marital_status"], 0.06)
    row["employment_status"] = corrupt(EMPLOY_V, row["employment_status"], 0.05)
    row["smoking_status"] = corrupt(SMOKE_V, row["smoking_status"], 0.05)
    row["alcohol_use"] = corrupt(ALCOHOL_V, row["alcohol_use"], 0.05)
    row["social_engagement"] = corrupt(SOCIAL_V, row["social_engagement"], 0.04)
    row["hiv_status"] = corrupt(HIV_V, row["hiv_status"], 0.03)

    # Whitespace padding on a few remaining text cells
    if bern(TEXT_PAD_PROB):
        col = random.choice(["social_engagement", "smoking_status", "employment_status"])
        row[col] = " %s " % str(row[col])
        cnt["whitespace_padding"] += 1

    # True NULLs (missingness)
    missing_income_p = 0.18 if row["residence"].strip().lower().startswith("rural") else 0.07
    null_plan = [
        ("monthly_household_income_usd", missing_income_p),
        ("alcohol_use", 0.06),
        ("gds15_score", 0.03),
        ("bmi_kg_m2", 0.04),
        ("systolic_bp_mmhg", 0.05),
        ("diastolic_bp_mmhg", 0.05),
        ("education_years", 0.02),
        ("hearing_difficulty", 0.02),
        ("vision_difficulty", 0.02),
        ("fruit_veg_servings_per_day", 0.03),
        ("social_engagement", 0.02),
        ("marital_status", 0.01),
        ("employment_status", 0.01),
        ("takes_bp_medication", 0.03),
        ("family_history_dementia", 0.02),
        ("head_injury_history", 0.02),
        ("cognitive_screen_score", 0.02),
    ]
    for col, p in null_plan:
        if bern(p):
            row[col] = None
            cnt["null_%s" % col] += 1
    if row["systolic_bp_mmhg"] is None:
        row["diastolic_bp_mmhg"] = None  # BP measured as a pair or not at all

    # Survey sentinel codes (-99 refusal, 88 don't know, 99 not applicable)
    sentinel_plan = [
        ("monthly_household_income_usd", 0.03, [88, 99]),
        ("education_years", 0.02, [-99]),
        ("gds15_score", 0.02, [-99, 88]),
        ("fruit_veg_servings_per_day", 0.02, [88, 99]),
    ]
    for col, p, opts in sentinel_plan:
        if isinstance(row[col], (int, float)) and bern(p):
            row[col] = random.choice(opts)
            cnt["sentinel_%s" % col] += 1

    # HIV unknown statuses (refusal/never tested)
    if bern(0.07):
        row["hiv_status"] = "Unknown"

# --- Targeted structural corruptions ---------------------------------------

n_rows = len(rows)
NOISE_MULT = max(1, N // 1500)  # keep engineered-defect densities constant as N grows


def sample_idx(k, exclude=frozenset()):
    pool = [j for j in range(n_rows) if j not in exclude]
    return set(random.sample(pool, k))


used = set()

# Impossible ages
bad_ages = [999, 145, 120, 150, 888, -5, 160, 777] * NOISE_MULT
idx_age = sample_idx(len(bad_ages), used); used |= idx_age
for j, v in zip(sorted(idx_age), bad_ages):
    rows[j]["age_years"] = v
cnt["impossible_age"] = len(bad_ages)

# Impossible BMI
bad_bmi = [78.4, 9.1, 102.5, 4.7, 65.0, 8.8] * NOISE_MULT
idx_bmi = sample_idx(len(bad_bmi), used); used |= idx_bmi
for j, v in zip(sorted(idx_bmi), bad_bmi):
    rows[j]["bmi_kg_m2"] = v
cnt["impossible_bmi"] = len(bad_bmi)

# Transposed blood pressure (diastolic >= systolic)
idx_bp = sample_idx(7 * NOISE_MULT, used); used |= idx_bp
for j in sorted(idx_bp):
    s, d2 = rows[j]["systolic_bp_mmhg"], rows[j]["diastolic_bp_mmhg"]
    if s is not None and d2 is not None:
        rows[j]["systolic_bp_mmhg"], rows[j]["diastolic_bp_mmhg"] = d2, s
cnt["bp_transposed"] = len(idx_bp)

# Impossible education years
bad_edu = [44, 39, 51, 27] * NOISE_MULT
idx_edu = sample_idx(len(bad_edu), used); used |= idx_edu
for j, v in zip(sorted(idx_edu), bad_edu):
    rows[j]["education_years"] = v
cnt["impossible_education"] = len(bad_edu)

# Impossible GDS scores (>15)
bad_gds = [27, 19, 23] * NOISE_MULT
idx_gds = sample_idx(len(bad_gds), used); used |= idx_gds
for j, v in zip(sorted(idx_gds), bad_gds):
    rows[j]["gds15_score"] = v
cnt["impossible_gds"] = len(bad_gds)

# Impossible fruit/veg servings
bad_fv = [-3, 45] * NOISE_MULT
idx_fv = sample_idx(len(bad_fv), used); used |= idx_fv
for j, v in zip(sorted(idx_fv), bad_fv):
    rows[j]["fruit_veg_servings_per_day"] = v
cnt["impossible_fruitveg"] = len(bad_fv)

# Logical contradictions: on BP medication but hypertension_dx = No
cand = [j for j in range(n_rows)
        if rows[j]["takes_bp_medication"] == 1 and rows[j]["hypertension_dx"] == 1]
idx_contra = set(random.sample(cand, min(10 * NOISE_MULT, len(cand))))
for j in idx_contra:
    rows[j]["hypertension_dx"] = 0
cnt["logic_contradiction"] = len(idx_contra)

# Inclusion violations: age below the >=60 protocol threshold
idx_under = sample_idx(6 * NOISE_MULT, used); used |= idx_under
for k, j in enumerate(sorted(idx_under)):
    rows[j]["age_years"] = 48 + k  # 48..53
cnt["inclusion_violation"] = len(idx_under)

# Duplicates: exact copies and fuzzy near-duplicates (same participant_id)
dup_exact = random.sample(range(n_rows), 15 * NOISE_MULT)
for j in dup_exact:
    rows.append(dict(rows[j]))
cnt["duplicate_exact"] = len(dup_exact)

dup_fuzzy = random.sample(range(n_rows), 6 * NOISE_MULT)
for j in dup_fuzzy:
    r2 = dict(rows[j])
    if isinstance(r2["age_years"], int):
        r2["age_years"] += random.choice([-1, 1])
    inc = r2["monthly_household_income_usd"]
    if isinstance(inc, float):
        r2["monthly_household_income_usd"] = inc + random.choice([-5.0, 5.0])
    rows.append(r2)
cnt["duplicate_fuzzy"] = len(dup_fuzzy)

total_inserted = len(rows)

# ----------------------------------------------------------------------------
# SQL emission
# ----------------------------------------------------------------------------

COLS = [
    "participant_id", "interview_date", "province", "residence", "sex",
    "age_years", "marital_status", "education_years", "employment_status",
    "monthly_household_income_usd", "lives_alone", "social_engagement",
    "hiv_status", "hypertension_dx", "takes_bp_medication", "diabetes_dx",
    "prior_stroke", "head_injury_history", "family_history_dementia",
    "systolic_bp_mmhg", "diastolic_bp_mmhg", "bmi_kg_m2", "gds15_score",
    "hearing_difficulty", "vision_difficulty", "smoking_status", "alcohol_use",
    "physical_activity_level", "fruit_veg_servings_per_day",
    "cognitive_screen_score", "dementia_status",
]


def sqlval(v):
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float):
        return repr(round(v, 2))
    return str(int(v))


L = []
A = L.append

A("-- ============================================================================")
A("-- data.sql -- SYNTHETIC dementia screening dataset (Zimbabwe calibration)")
A("-- Generated by generate_data.py (seed=42). Regenerate: python generate_data.py")
A("-- ============================================================================")
A("--")
A("-- PROVENANCE / ETHICS DISCLAIMER")
A("--   This dataset is 100% SYNTHETIC. It contains NO real patient records.")
A("--   It was generated procedurally so that marginal distributions are")
A("--   plausible for a community sample of adults aged >=60 in Zimbabwe.")
A("--   Any analysis results obtained from it are ILLUSTRATIVE ONLY and must")
A("--   never be quoted as findings about Zimbabwe.")
A("--")
A("-- CALIBRATION ANCHORS (approximate, published sources)")
A("--   Age structure, 10-province weights, ~30% urban share .... ZIMSTAT,")
A("--       2022 Population and Housing Census of Zimbabwe")
A("--   Overall dementia prevalence target ~7.8% (rising with age) ... pooled")
A("--       Sub-Saharan Africa estimates (Guerchet et al.; HAALSI/de Jager et al.,")
A("--       Neurology 2017, rural South Africa)")
A("--   Hypertension common among 60+, treatment coverage <50% ... WHO STEPS")
A("--       Zimbabwe 2014; SSA elderly reviews")
A("--   Diabetes ~5-10% adults ... STEPS 2014 / IDF SSA region")
A("--   Adult HIV prevalence ~11-12%; assumed 3-6% among 60+ (ASSUMPTION,")
A("--       flagged) ... UNAIDS AIDSinfo country profile: Zimbabwe")
A("--   Smoking male-skewed (~15% men, ~1-2% women) ... STEPS 2014")
A("--   High widowhood among elderly women; low schooling in oldest cohorts ...")
A("--       DHS/MICS attainment patterns; colonial-era access")
A("--   Low USD household incomes, informal economy ... World Bank/ZIMSTAT")
A("--")
A("-- ENGINE COMPATIBILITY")
A("--   Portable ANSI-style SQL (INTEGER/REAL/TEXT only, explicit IDs).")
A("--   Tested target: SQLite. Also loads on PostgreSQL and MySQL.")
A("--   NOTE: interview_date is TEXT because three date formats coexist")
A("--   deliberately; strict DATE columns would reject some engines' inserts.")
A("--   No PRIMARY KEY constraint on participant_id: exact duplicates exist")
A("--   by design (cleaning exercise).")
A("--")
A("-- IMPORT")
A("--   SQLite : sqlite3 dementia.db < data.sql")
A("--   psql   : psql -f data.sql")
A("--   MySQL  : mysql -e \"SOURCE data.sql\"  (or pipe it in)")
A("--")
A("-- CONTENTS")
A("--   dementia_screening : %d inserted rows (%d unique participants)" % (total_inserted, N))
A("--   data_dictionary    : column definitions")
A("--   noise_manifest     : engineered defects + cleaning recommendations")
A("--   Suggested QA queries at bottom of file.")
A("-- ============================================================================")
A("")
A("BEGIN TRANSACTION;")
A("")
A("DROP TABLE IF EXISTS dementia_screening;")
A("DROP TABLE IF EXISTS data_dictionary;")
A("DROP TABLE IF EXISTS noise_manifest;")
A("")
A("CREATE TABLE dementia_screening (")
A("    participant_id                  TEXT NOT NULL,")
A("    interview_date                  TEXT,           -- mixed formats, see noise_manifest")
A("    province                        TEXT,           -- 10 provinces")
A("    residence                       TEXT,           -- Urban/Rural (dirty variants)")
A("    sex                             TEXT,           -- M/F (dirty variants)")
A("    age_years                       INTEGER,        -- protocol requires >= 60")
A("    marital_status                  TEXT,")
A("    education_years                 INTEGER,        -- completed formal schooling")
A("    employment_status               TEXT,")
A("    monthly_household_income_usd    REAL,")
A("    lives_alone                     INTEGER,        -- 0/1")
A("    social_engagement               TEXT,           -- Daily/Weekly/Rarely/Never")
A("    hiv_status                      TEXT,           -- Positive/Negative/Unknown")
A("    hypertension_dx                 INTEGER,        -- 0/1 self-reported diagnosis")
A("    takes_bp_medication             INTEGER,        -- 0/1 current BP medication")
A("    diabetes_dx                     INTEGER,        -- 0/1")
A("    prior_stroke                    INTEGER,        -- 0/1")
A("    head_injury_history             INTEGER,        -- 0/1 significant TBI history")
A("    family_history_dementia         INTEGER,        -- 0/1 first-degree relative")
A("    systolic_bp_mmhg                INTEGER,        -- mean of 2 seated readings")
A("    diastolic_bp_mmhg               INTEGER,")
A("    bmi_kg_m2                       REAL,")
A("    gds15_score                     INTEGER,        -- Geriatric Depression Scale 0-15")
A("    hearing_difficulty              INTEGER,        -- 0/1 self-reported")
A("    vision_difficulty               INTEGER,        -- 0/1 self-reported")
A("    smoking_status                  TEXT,           -- Never/Former/Current (dirty)")
A("    alcohol_use                     TEXT,           -- None/Moderate/Hazardous (dirty)")
A("    physical_activity_level         TEXT,           -- Low/Moderate/High")
A("    fruit_veg_servings_per_day      INTEGER,")
A("    cognitive_screen_score          INTEGER,        -- 0-30 MMSE/CSI-D-style screen;")
A("                                                    -- OUTCOME ASCERTAINMENT VARIABLE:")
A("                                                    -- EXCLUDE from predictors (leakage)")
A("    dementia_status                 INTEGER NOT NULL -- DEPENDENT VARIABLE 0/1,")
A("                                                     -- clinician-consensus gold standard")
A(");")
A("")
A("-- ---------------------------------------------------------------------------")
A("-- Main table inserts (%d rows; %d unique participant_id values)" % (total_inserted, N))
A("-- ---------------------------------------------------------------------------")

for row in rows:
    vals = ", ".join(sqlval(row[c]) for c in COLS)
    A("INSERT INTO dementia_screening (%s) VALUES (%s);" % (", ".join(COLS), vals))

A("")
A("-- ---------------------------------------------------------------------------")
A("-- Data dictionary")
A("-- ---------------------------------------------------------------------------")
A("CREATE TABLE data_dictionary (")
A("    column_name      TEXT NOT NULL,")
A("    data_type        TEXT,")
A("    allowed_values   TEXT,")
A("    description      TEXT")
A(");")

DICT_ROWS = [
    ("participant_id", "TEXT", "'ZW-#####'", "Pseudonymised participant identifier. NOT unique in raw table (duplicates by design)."),
    ("interview_date", "TEXT", "3 formats: ISO, DD/MM/YYYY, 'Mon D, YYYY'", "Date of Phase-1 interview within 2024-01-08..2025-03-31 fieldwork window."),
    ("province", "TEXT", "10 Zimbabwe provinces (+dirty variants)", "Province of residence."),
    ("residence", "Text", "Urban/Rural (+dirty variants)", "Urban or rural residence."),
    ("sex", "TEXT", "M/F (+dirty variants)", "Sex of participant."),
    ("age_years", "INTEGER", ">=60 per protocol (violations exist)", "Age in completed years at interview."),
    ("marital_status", "TEXT", "Married/Widowed/Divorced-Separated/Never Married (+variants)", "Current marital status."),
    ("education_years", "INTEGER", "0-16 (+impossible values)", "Completed years of formal schooling."),
    ("employment_status", "TEXT", "Retired/Subsistence Farming/Informal Trading/Formal Employment/Unemployed (+synonyms)", "Main current occupation."),
    ("monthly_household_income_usd", "REAL", ">0 (+88/99 sentinels, NULL)", "Self-reported household income, USD per month."),
    ("lives_alone", "INTEGER", "0/1", "Lives alone in household."),
    ("social_engagement", "TEXT", "Daily/Weekly/Rarely/Never (+variants)", "Frequency of social contact outside household."),
    ("hiv_status", "TEXT", "Positive/Negative/Unknown", "Self-reported HIV status; Unknown includes never-tested/refusal."),
    ("hypertension_dx", "INTEGER", "0/1", "Ever told by health worker they have high blood pressure."),
    ("takes_bp_medication", "INTEGER", "0/1 (+NULL)", "Currently taking blood-pressure medication."),
    ("diabetes_dx", "INTEGER", "0/1", "Ever told they have diabetes."),
    ("prior_stroke", "INTEGER", "0/1", "Self/family-reported prior stroke episode."),
    ("head_injury_history", "INTEGER", "0/1", "History of significant head injury with loss of consciousness."),
    ("family_history_dementia", "INTEGER", "0/1", "Dementia in a first-degree relative."),
    ("systolic_bp_mmhg", "INTEGER", "~80-260 (+NULL, transpositions)", "Mean seated systolic BP, mmHg."),
    ("diastolic_bp_mmhg", "INTEGER", "~45-140 (+NULL, transpositions)", "Mean seated diastolic BP, mmHg."),
    ("bmi_kg_m2", "REAL", "~13.5-45 (+impossible values)", "Body mass index, kg/m^2."),
    ("gds15_score", "INTEGER", "0-15 (+sentinels, impossible values)", "Geriatric Depression Scale short-form score."),
    ("hearing_difficulty", "INTEGER", "0/1", "Self-reported difficulty hearing (with aid if owned)."),
    ("vision_difficulty", "INTEGER", "0/1", "Self-reported difficulty seeing (with aid if owned)."),
    ("smoking_status", "TEXT", "Never/Former/Current (+synonyms like 'Ex-smoker', ambiguous 'Yes')", "Tobacco smoking status."),
    ("alcohol_use", "TEXT", "None/Moderate/Hazardous (+synonyms)", "Alcohol use category (AUDIT-C style grouping)."),
    ("physical_activity_level", "TEXT", "Low/Moderate/High", "IPAQ-style composite physical activity level."),
    ("fruit_veg_servings_per_day", "INTEGER", "0-8 (+sentinels, impossible values)", "Usual daily servings of fruit and/or vegetables."),
    ("cognitive_screen_score", "INTEGER", "0-30 (+NULL)", "MMSE/CSI-D-style composite screen. USED IN OUTCOME ASCERTAINMENT - do NOT use as predictor (data leakage)."),
    ("dementia_status", "INTEGER", "0/1", "DEPENDENT VARIABLE. Clinician-consensus dementia diagnosis (gold-standard simulation). Never NULL."),
]
for dr in DICT_ROWS:
    A("INSERT INTO data_dictionary VALUES (%s);" %
      ", ".join(sqlval(x) for x in dr))

A("")
A("-- ---------------------------------------------------------------------------")
A("-- Noise manifest: engineered defects + recommended cleaning actions")
A("-- ---------------------------------------------------------------------------")
A("CREATE TABLE noise_manifest (")
A("    noise_id                INTEGER NOT NULL,")
A("    noise_category          TEXT,")
A("    affected_columns        TEXT,")
A("    approx_count            TEXT,")
A("    example_value           TEXT,")
A("    cleaning_recommendation TEXT")
A(");")


def cfmt(key, suffix="cells"):
    return "~%d %s" % (cnt.get(key, 0), suffix)


NOISE_ROWS = [
    (1, "Missingness MCAR", "many numeric/categorical columns", "~2-6% per column", "NULL",
     "Profile missingness per column; impute AFTER train/test split (median/mode fitted on train only) or complete-case sensitivity analysis."),
    (2, "Missingness MAR", "monthly_household_income_usd", "~18% rural vs ~7% urban", "NULL",
     "Income under-reporting concentrated in rural households; consider missing-indicator flag plus imputation."),
    (3, "Survey sentinel codes", "income, education, GDS, fruit/veg", cfmt("sentinel_monthly_household_income_usd") + "+", "-99 / 88 / 99",
     "Map -99=Refusal, 88=Don't know, 99=Not applicable -> NULL before any arithmetic; NEVER treat as valid numbers."),
    (4, "Exact duplicates", "whole rows", "%d rows" % cnt.get("duplicate_exact", 0), "same participant_id twice",
     "Detect via GROUP BY participant_id HAVING COUNT(*)>1; keep one; document conflict rule."),
    (5, "Fuzzy duplicates", "whole rows", "%d rows" % cnt.get("duplicate_fuzzy", 0), "same id, age +/-1",
     "Near-duplicates with tiny jitter; deduplicate on participant_id keeping first occurrence."),
    (6, "Impossible values", "age_years", cfmt("impossible_age"), "999, -5, 160",
     "Range-check against protocol (>=60, plausibility cap ~105); route to rejection log, do not silently overwrite."),
    (7, "Impossible values", "bmi_kg_m2", cfmt("impossible_bmi"), "78.4, 4.7",
     "Physiologic bounds ~10-70; set NULL and flag."),
    (8, "Unit/transposition errors", "systolic_bp_mmhg, diastolic_bp_mmhg", cfmt("bp_transposed", "rows"), "dia >= sys",
     "Where diastolic >= systolic assume transposition; swap then validate; else reject."),
    (9, "Impossible values", "education_years, gds15_score, fruit_veg_servings_per_day", cfmt("impossible_education"), "44 school years; GDS 27; -3 servings",
     "Domain range checks (edu<=16~20, GDS 0-15, servings>=0)."),
    (10, "Category inconsistencies", "sex, province, residence, marital, employment, smoking, alcohol, social, HIV", cfmt("category_variants"), "'MALE ', 'widwoed', 'Hre'",
     "Build controlled vocabularies + mapping tables; trim whitespace; casefold; fuzzy-match misspellings."),
    (11, "Whitespace padding", "several text columns", cfmt("whitespace_padding"), "' Vendor '",
     "TRIM all text fields early in pipeline."),
    (12, "Date format inconsistency", "interview_date", "3 formats across file", "'15/03/2024' vs '2024-03-15' vs 'Mar 15, 2024'",
     "Parse with explicit multi-format parser; fail loudly on unparseable dates."),
    (13, "Logical contradictions", "hypertension_dx vs takes_bp_medication", cfmt("logic_contradiction", "rows"), "medication=1 but dx=0",
     "Flag contradictions; decide keep-as-missing vs keep-with-flag; document choice."),
    (14, "Inclusion violations", "age_years", cfmt("inclusion_violation", "rows"), "age 48-53",
     "Apply eligibility filter age>=60 after cleaning; report excluded count."),
]
for nr in NOISE_ROWS:
    A("INSERT INTO noise_manifest VALUES (%s);" % ", ".join(sqlval(x) for x in nr))

A("")
A("COMMIT;")
A("")
A("-- ---------------------------------------------------------------------------")
A("-- Suggested post-import QA queries (run manually)")
A("-- ---------------------------------------------------------------------------")
A("-- Row counts and duplicates:")
A("--   SELECT COUNT(*) AS rows_total, COUNT(DISTINCT participant_id) AS uniq FROM dementia_screening;")
A("-- Crude prevalence among protocol-eligible unique participants:")
A("--   SELECT AVG(dementia_status) FROM dementia_screening WHERE age_years BETWEEN 60 AND 110;")
A("--   (deduplicate first!)")
A("-- Distinct dirty categories:")
A("--   SELECT DISTINCT sex FROM dementia_screening ORDER BY sex;")
A("--   SELECT DISTINCT interview_date LIKE '%/%' FROM dementia_screening LIMIT 5;")
A("-- Sentinel scan:")
A("--   SELECT COUNT(*) FROM dementia_screening WHERE gds15_score IN (-99, 88, 99);")
A("")
A("-- End of data.sql")

sql_text = "\n".join(L) + "\n"
with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
    f.write(sql_text)

# ----------------------------------------------------------------------------
# CSV emission (flat mirror of dementia_screening; NULL -> empty field)
# ----------------------------------------------------------------------------

with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    wtr = csv.writer(f, lineterminator="\n")
    wtr.writerow(COLS)
    for row in rows:
        wtr.writerow(["" if row[c] is None else row[c] for c in COLS])

# ----------------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------------
print("Wrote %s" % OUT_PATH)
print("Wrote %s (%d data rows + header)" % (CSV_PATH, total_inserted))
print("Inserted rows      : %d" % total_inserted)
print("Unique participants: %d" % N)
print("Achieved prevalence: %.3f (target %.3f)" % (achieved_prev, TARGET_PREVALENCE))
print("Intercept (logit)  : %.3f" % INTERCEPT)
events = sum(r["dementia_status"] for r in rows)
print("Events (cases)     : %d" % events)
print("Noise counter top items:")
for k, v in cnt.most_common(12):
    print("   %-38s %d" % (k, v))
