"""Build `data1.csv` — a class-enriched, noise-preserving derivative of `data.csv`.

Why: `data.csv` is synthetic data modelled to mimic real screening data, but only ~8% of
rows are dementia (+) cases. That imbalance trains models that are good at identifying
(-) cases and weak on (+) cases. This script creates a SEPARATE file (the original
`data.csv` is never modified) that:

  1. keeps every original record as-is (with all its engineered noise),
  2. synthesises additional (+) records by jittering the (+)-class donor pool so that
     positives make up >= 31.9% of the file (target ~32.5%),
  3. injects extra realistic noise on top: missing values, survey sentinel codes,
     date-format chaos, case/whitespace variants, BP transpositions, logical
     contradictions, exact + fuzzy duplicates, impossible values and a few
     protocol-violating under-60 ages — the same defect families the cleaning
     pipeline in `modelx.ipynb` knows how to handle.

Deterministic: seeded RNG (42), so `data1.csv` is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
TARGET_PREVALENCE = 0.325          # raw-file target; hard floor is 0.319
MIN_PREVALENCE = 0.319

# ----------------------------------------------------------------------------- load
raw = pd.read_csv("data.csv", keep_default_na=False, dtype=str)   # "" stays "", mess preserved
N_ORIG = len(raw)
pos_mask = raw["dementia_status"] == "1"
n_pos = int(pos_mask.sum())
n_new = int(np.ceil((TARGET_PREVALENCE * N_ORIG - n_pos) / (1 - TARGET_PREVALENCE)))
print(f"data.csv: {N_ORIG:,} rows | positives {n_pos:,} ({n_pos / N_ORIG:.2%})")
print(f"synthesising {n_new:,} extra (+) records -> target prevalence "
      f"{(n_pos + n_new) / (N_ORIG + n_new):.2%}")

CAT_COLS = ["province", "residence", "sex", "marital_status", "employment_status",
            "hiv_status", "smoking_status", "alcohol_use", "physical_activity_level",
            "social_engagement"]
BIN_COLS = ["lives_alone", "hypertension_dx", "takes_bp_medication", "diabetes_dx",
            "prior_stroke", "head_injury_history", "family_history_dementia",
            "hearing_difficulty", "vision_difficulty"]
# numeric jitter spec: col -> (round_digits, jitter_sd_fraction_of_sd, lo, hi)
NUM_JITTER = {
    "education_years":               (0, 0.35, 0, 20),
    "monthly_household_income_usd":  (1, 0.35, 0, 800),
    "systolic_bp_mmhg":              (0, 0.08, 70, 250),
    "diastolic_bp_mmhg":             (0, 0.08, 40, 150),
    "bmi_kg_m2":                     (1, 0.12, 10, 70),
    "gds15_score":                   (0, 0.30, 0, 15),
    "fruit_veg_servings_per_day":    (0, 0.35, 0, 20),
    "cognitive_screen_score":        (0, 0.10, 0, 30),
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]

# (+)-class reference distributions, restricted to plausible values so that jitter
# statistics are not polluted by the engineered defects.
donors = raw[pos_mask].reset_index(drop=True)
plaus_num = {}
for col, (_, _, lo, hi) in NUM_JITTER.items():
    v = pd.to_numeric(donors[col], errors="coerce")
    plaus_num[col] = v[v.between(lo, hi)]
plaus_age = pd.to_numeric(donors["age_years"], errors="coerce")
plaus_age = plaus_age[plaus_age.between(60, 100)]


def sample_marginal(series, k=1):
    """Draw k raw labels (mess variants included) from the (+)-class empirical distribution."""
    vals = series.dropna()
    vals = vals[vals != ""]
    counts = vals.value_counts()
    return RNG.choice(counts.index.to_numpy(), size=k, p=counts.values / counts.values.sum())


def fmt(col, v):
    digits = NUM_JITTER[col][0]
    return f"{v:.1f}" if digits == 1 else str(int(round(v)))


def jitter_numeric(row, col):
    """Donor value + Gaussian jitter; missing/implausible donor cells resample from the
    (+)-class plausible marginal, and a little extra missingness is left in place."""
    (_, frac, lo, hi) = NUM_JITTER[col]
    if RNG.random() < 0.03:                       # keep a little missingness
        return ""
    v = pd.to_numeric(row[col], errors="coerce")
    if pd.isna(v) or not (lo <= v <= hi):
        v = float(RNG.choice(plaus_num[col].to_numpy()))
    else:
        sd = float(plaus_num[col].std())
        v = v + RNG.normal(0.0, max(frac * sd, 1e-6))
    return fmt(col, float(np.clip(v, lo, hi)))


def synth_row(i):
    """One new (+) record: (+)-donor numerics with jitter, (+)-marginal categoricals."""
    donor = donors.iloc[int(RNG.integers(len(donors)))]
    row = {c: "" for c in raw.columns}
    # age: jitter a plausible donor age; repair implausible / under-60 donors by resampling
    if RNG.random() < 0.02:
        age = ""
    else:
        a = pd.to_numeric(donor["age_years"], errors="coerce")
        if pd.isna(a) or not (60 <= a <= 100):
            a = float(RNG.choice(plaus_age.to_numpy()))
        a = float(np.clip(a + RNG.normal(0, 2.5), 60, 100))
        age = str(int(round(a)))
    row["age_years"] = age
    for col in NUM_JITTER:
        row[col] = jitter_numeric(donor, col)
    for col in CAT_COLS + BIN_COLS:
        row[col] = str(sample_marginal(donors[col], 1)[0])
    row["dementia_status"] = "1"
    ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(RNG.integers(0, 456)))
    row["interview_date"] = ts.strftime(DATE_FORMATS[int(RNG.integers(len(DATE_FORMATS)))])
    row["participant_id"] = f"ZW-{12_001 + i:05d}"
    return row


synth = pd.DataFrame([synth_row(i) for i in range(n_new)])
data1 = pd.concat([raw, synth], ignore_index=True)

# ---------------------------------------------------------------- noise injection
def random_rows(n, mask=None):
    pool = data1.index if mask is None else data1.index[mask]
    return RNG.choice(pool, size=n, replace=False)


# (1) extra true missing values (empty fields) across survey-prone columns
for col, rate in [("monthly_household_income_usd", 0.010), ("education_years", 0.008),
                  ("bmi_kg_m2", 0.006), ("gds15_score", 0.006), ("systolic_bp_mmhg", 0.005),
                  ("diastolic_bp_mmhg", 0.005), ("fruit_veg_servings_per_day", 0.006),
                  ("cognitive_screen_score", 0.005), ("alcohol_use", 0.006),
                  ("social_engagement", 0.005), ("takes_bp_medication", 0.004)]:
    idx = random_rows(int(rate * len(data1)), data1[col] != "")
    data1.loc[idx, col] = ""

# (2) survey sentinel codes (-99 refusal / 88 don't know / 99 not applicable)
for col in ["monthly_household_income_usd", "education_years", "gds15_score",
            "fruit_veg_servings_per_day"]:
    idx = random_rows(int(0.004 * len(data1)), data1[col] != "")
    data1.loc[idx, col] = RNG.choice(["-99", "88", "99"], size=len(idx))

# (3) case / whitespace / typo chaos on categoricals (same defect families as data.csv)
TYPO = {"Harare": "hre", "Rural": "Rual", "Urban": "Urbn", "Married": "maried",
        "Widowed": "widwoed", "Divorced/Separated": "divorced/seperated",
        "Never Married": "single", "Current": "CURRENT SMOKER", "Never": "never smoked",
        "Subsistence Farming": "Farming", "None": "no", "Positive": "POS"}
def chaos(v):
    muts = [v.lower(), v.upper(), " " + v, v + " ", v.replace(" ", "  ")]
    if v in TYPO:
        muts.append(TYPO[v])
    return str(RNG.choice(muts))
for col in CAT_COLS:
    idx = random_rows(int(0.008 * len(data1)), data1[col] != "")
    data1.loc[idx, col] = data1.loc[idx, col].map(chaos)

# (4) BP transpositions (diastolic recorded >= systolic)
both = data1["systolic_bp_mmhg"].ne("") & data1["diastolic_bp_mmhg"].ne("")
idx = random_rows(45, both)
sys_v, dia_v = data1.loc[idx, "systolic_bp_mmhg"].copy(), data1.loc[idx, "diastolic_bp_mmhg"].copy()
data1.loc[idx, "systolic_bp_mmhg"], data1.loc[idx, "diastolic_bp_mmhg"] = dia_v, sys_v

# (5) logical contradictions: BP medication claimed with no hypertension diagnosis
contra = data1.index[data1["hypertension_dx"] == "0"]
idx = RNG.choice(contra, size=45, replace=False)
data1.loc[idx, "takes_bp_medication"] = "1"

# (6) exact duplicates (all fields) + fuzzy duplicates (same participant_id, age jitter)
dup_exact = data1.loc[RNG.choice(data1.index[-n_new:], size=60, replace=False)].copy()
fuzzy_base = data1.loc[RNG.choice(data1.index[-n_new:], size=20, replace=False)].copy()
fuzzy_base["age_years"] = [
    str(int(pd.to_numeric(a) + RNG.choice([-1, 1]))) for a in fuzzy_base["age_years"]]
data1 = pd.concat([data1, dup_exact, fuzzy_base], ignore_index=True)

# (7) impossible / out-of-range values (repaired -> NaN or rejected by cleaning)
def set_vals(col, values):
    idx = random_rows(len(values), data1[col] != "")
    data1.loc[idx, col] = values
set_vals("age_years", ["999", "-5", "160", "999", "-5", "160", "999", "888"])
set_vals("education_years", ["51", "51", "39", "51"])
set_vals("bmi_kg_m2", ["4.7", "102.5", "90.3", "4.7", "102.5"])
set_vals("gds15_score", ["150", "150", "27", "39"])
set_vals("fruit_veg_servings_per_day", ["-3", "99", "-3"])

# (8) a few under-60 protocol violations among the new (+) records
u60 = RNG.choice(data1.index[-n_new:], size=8, replace=False)
data1.loc[u60, "age_years"] = [str(int(RNG.integers(48, 54))) for _ in u60]

# ---------------------------------------------------------------- finalise
data1 = data1.iloc[RNG.permutation(len(data1))].reset_index(drop=True)
prev = (data1["dementia_status"] == "1").mean()
assert prev >= MIN_PREVALENCE, f"prevalence {prev:.2%} below required {MIN_PREVALENCE:.1%}"
data1.to_csv("data1.csv", index=False)

miss = (data1 == "").mean().mul(100).sort_values(ascending=False).head(8)
print(f"data1.csv written: {len(data1):,} rows | positives {int((data1['dementia_status'] == '1').sum()):,} "
      f"({prev:.2%})  [floor {MIN_PREVALENCE:.1%}]")
print(f"noise injected: missing cells ~{sum(r for c, r in [('a', 0.010), ('b', 0.008), ('c', 0.006), ('d', 0.006), ('e', 0.005), ('f', 0.005), ('g', 0.006), ('h', 0.005), ('i', 0.006), ('j', 0.005), ('k', 0.004)]):.1%} of rows | 180 sentinel cells | 360 chaos labels | 45 BP swaps | 45 contradictions | 80 duplicates | 30 impossible values | 8 under-60")
print("top missingness now:")
print(miss.round(2).to_string())
print("data.csv untouched (this script only reads it and writes data1.csv)")
