from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths / Config
# ============================================================

# Assumes this file is in: <project_root>/src/cw2_main.py
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

COUNTRY_E = "Japan"                    # for (e)
COUNTRY_H = "Japan"                    # for (h)
COUNTRIES_F = ("Japan", "France")     # for (f)
COUNTRIES_J = ("Japan", "Ukraine", "France")  # for (j)

FILE_COUNTRY = "countries_data.csv"
FILE_EDU = "world_bank_education.csv"


# ============================================================
# Helpers
# ============================================================

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_country_name(name: object) -> str:
    """
    Normalize country names for matching across datasets.
    Keeps it simple and transparent (no fuzzy matching).
    """
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\,']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def find_year_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect year columns robustly.

    Accepts common World Bank export formats:
      - "2010 [YR2010]"
      - "2010[YR2010]"
      - "2010"
    Also tolerates extra spaces.
    """
    cols = [str(c).strip() for c in df.columns]
    pattern = re.compile(r"^(19|20)\d{2}(\s*\[\s*YR(19|20)\d{2}\s*\])?$")
    year_cols = [c for c in cols if pattern.match(c)]

    # Some exports become pure integers -> cast to str already handled,
    # but keep an extra fallback just in case.
    if not year_cols:
        year_cols = [c for c in cols if re.fullmatch(r"(19|20)\d{2}", c)]

    return year_cols


def save_plot(fig: plt.Figure, filename: str) -> Path:
    out_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def get_indicator(df_long: pd.DataFrame, keywords: Iterable[str]) -> pd.DataFrame:
    """
    Filter long-format education data for rows whose indicator contains ALL keywords.
    Case-insensitive. Returns a copy.
    """
    ind = df_long["indicator"].astype(str).str.lower()
    mask = pd.Series(True, index=df_long.index)
    for kw in keywords:
        mask &= ind.str.contains(str(kw).lower(), na=False)
    return df_long.loc[mask].copy()


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce World Bank-ish values to numeric robustly.
    Handles common missing markers and stray whitespace.
    """
    # Avoid pandas FutureWarning by NOT mixing types in replace on the whole Series.
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "..": np.nan, "nan": np.nan, "NaN": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def classify_density(d: float) -> str:
    if pd.isna(d):
        return "unknown"
    if d > 500:
        return "very_dense"
    if 100 <= d <= 500:
        return "dense"
    return "sparse"


# ============================================================
# (a) Load + name mismatches
# ============================================================

def task_a_load_and_compare() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    df_country = pd.read_csv(DATA_DIR / FILE_COUNTRY)
    df_education = pd.read_csv(DATA_DIR / FILE_EDU)

    # Strip column names to avoid invisible trailing spaces
    df_country.columns = [c.strip() for c in df_country.columns]
    df_education.columns = [c.strip() for c in df_education.columns]

    # Create normalized country fields
    # countries_data.csv has "country"
    df_country["country_norm"] = df_country["country"].map(normalize_country_name)

    # education has "Country Name"
    df_education["country_norm"] = df_education["Country Name"].map(normalize_country_name)

    set_country = set(df_country["country_norm"].dropna().unique())
    set_edu = set(df_education["country_norm"].dropna().unique())

    in_country_not_edu = sorted(list(set_country - set_edu))
    in_edu_not_country = sorted(list(set_edu - set_country))

    return df_country, df_education, in_country_not_edu, in_edu_not_country


# ============================================================
# (b) Filter education to common countries
# ============================================================

def task_b_filter_common(df_country: pd.DataFrame, df_education: pd.DataFrame) -> pd.DataFrame:
    common = set(df_country["country_norm"].dropna().unique()) & set(df_education["country_norm"].dropna().unique())
    df_education_filtered = df_education.loc[df_education["country_norm"].isin(common)].copy()
    return df_education_filtered


# ============================================================
# (c) Clean + transform to long format
# ============================================================

def task_c_clean_transform(df_education_filtered: pd.DataFrame) -> pd.DataFrame:

    year_cols = find_year_columns(df_education_filtered)
    if not year_cols:
        raise ValueError(
            "No year columns detected in education dataset.\n"
            "Open the CSV and check how the year columns are named (e.g., '2010 [YR2010]' or '2010')."
        )

    base_cols = ["Country Name", "Country Code", "Series", "Series Code", "country_norm"]
    missing = [c for c in base_cols if c not in df_education_filtered.columns]
    if missing:
        raise ValueError(f"Missing expected columns in education dataset: {missing}")

    df = df_education_filtered.copy()

    # Deduplicate exact duplicates
    df = df.drop_duplicates()

    # Melt wide -> long
    df_long = df.melt(
        id_vars=base_cols,
        value_vars=year_cols,
        var_name="year_raw",
        value_name="value_raw",
    )

    # Extract year as int from year_raw
    df_long["year"] = df_long["year_raw"].astype(str).str.extract(r"((19|20)\d{2})", expand=False)[0]
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")

    # Value numeric
    df_long["value"] = coerce_numeric_series(df_long["value_raw"])

    # Clean/rename columns
    df_long = df_long.rename(
        columns={
            "Country Name": "country_name",
            "Country Code": "country_code",
            "Series": "indicator",
            "Series Code": "indicator_code",
        }
    )

    # Keep only relevant columns
    df_long = df_long[["country_name", "country_norm", "country_code", "indicator", "indicator_code", "year", "value"]]

    # Drop rows with no year
    df_long = df_long.dropna(subset=["year"])

    return df_long


# ============================================================
# (d) Country with the largest number of secondary teachers in 2019
# ============================================================

def task_d_max_secondary_teachers_2019(df_long: pd.DataFrame) -> pd.Series:
    sec_teachers = get_indicator(df_long, ["teachers", "secondary", "number"])
    df_2019 = sec_teachers.loc[sec_teachers["year"] == 2019].dropna(subset=["value"])
    if df_2019.empty:
        raise ValueError("No non-missing secondary teachers data for 2019 found.")
    return df_2019.loc[df_2019["value"].idxmax()]


# ============================================================
# (e) Plot government expenditure on secondary education over years for a country
# ============================================================

def task_e_plot_secondary_expenditure(df_long: pd.DataFrame, country: str) -> Path:
    exp_secondary = get_indicator(df_long, ["government expenditure", "secondary"])
    df_c = exp_secondary.loc[exp_secondary["country_name"].str.lower() == country.lower()].copy()
    df_c = df_c.dropna(subset=["value"]).sort_values("year")
    if df_c.empty:
        raise ValueError(f"No secondary expenditure data found for {country}.")

    fig = plt.figure()
    plt.plot(df_c["year"].astype(int), df_c["value"])
    plt.title(f"{country}: Government expenditure on secondary education over time")
    plt.xlabel("Year")
    plt.ylabel("Government expenditure on secondary education")
    return save_plot(fig, f"e_gov_expenditure_secondary_{normalize_country_name(country)}.png")


# ============================================================
# (f) Teachers-to-enrolment ratio for each level, each year
# ============================================================

def task_f_teacher_to_enrolment_ratio(df_long: pd.DataFrame) -> pd.DataFrame:
    teachers = get_indicator(df_long, ["teachers", "both sexes", "number"])
    enrol = get_indicator(df_long, ["enrolment", "both sexes", "number"])

    def level_from_indicator(s: str) -> str:
        s = str(s).lower()
        if "pre-primary" in s or "pre primary" in s:
            return "pre_primary"
        if "primary" in s and "pre" not in s:
            return "primary"
        if "secondary" in s:
            return "secondary"
        return "other"

    teachers = teachers.copy()
    enrol = enrol.copy()
    teachers["level"] = teachers["indicator"].map(level_from_indicator)
    enrol["level"] = enrol["indicator"].map(level_from_indicator)

    teachers = teachers.loc[teachers["level"].isin(["pre_primary", "primary", "secondary"])]
    enrol = enrol.loc[enrol["level"].isin(["pre_primary", "primary", "secondary"])]

    # Aggregate in case there are multiple rows per (country, year, level)
    t_agg = (
        teachers.groupby(["country_norm", "country_name", "country_code", "year", "level"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "teachers"})
    )
    e_agg = (
        enrol.groupby(["country_norm", "country_name", "country_code", "year", "level"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "enrolment"})
    )

    ratio = pd.merge(t_agg, e_agg, on=["country_norm", "country_name", "country_code", "year", "level"], how="outer")
    ratio["ratio"] = ratio["teachers"] / ratio["enrolment"]
    return ratio


def task_f_show_two_countries(ratio_df: pd.DataFrame, countries: tuple[str, str]) -> pd.DataFrame:
    c_norm = [normalize_country_name(c) for c in countries]
    out = ratio_df.loc[ratio_df["country_norm"].isin(c_norm)].copy()
    out = out.sort_values(["country_name", "level", "year"])
    return out


# ============================================================
# (g) Largest increase/decrease in primary enrolment in 2019 (vs 2018)
# ============================================================

def task_g_primary_enrolment_change_2019(df_long: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    enrol_primary = get_indicator(df_long, ["enrolment", "primary", "both sexes", "number"])
    enrol_primary = enrol_primary.copy()
    enrol_primary = enrol_primary.loc[~enrol_primary["indicator"].astype(str).str.lower().str.contains("pre-primary", na=False)]

    wide = enrol_primary.pivot_table(index=["country_norm", "country_name"], columns="year", values="value", aggfunc="mean")
    if 2018 not in wide.columns or 2019 not in wide.columns:
        raise ValueError("Need both 2018 and 2019 primary enrolment columns to compute change.")

    tmp = wide[[2018, 2019]].copy()
    tmp = tmp.dropna()
    tmp["delta"] = tmp[2019] - tmp[2018]
    tmp = tmp.reset_index().rename(columns={2018: "enrol_2018", 2019: "enrol_2019"})

    max_inc = tmp.loc[tmp["delta"].idxmax()]
    max_dec = tmp.loc[tmp["delta"].idxmin()]
    return max_inc, max_dec


# ============================================================
# (h) Ratio (exp primary)/(enrol primary) over years for a country
# ============================================================

def task_h_plot_exp_to_primary_enrol_ratio(df_long: pd.DataFrame, country: str) -> Path:
    exp_primary = get_indicator(df_long, ["government expenditure", "primary"])
    enrol_primary = get_indicator(df_long, ["enrolment", "primary", "both sexes", "number"])
    enrol_primary = enrol_primary.loc[~enrol_primary["indicator"].astype(str).str.lower().str.contains("pre-primary", na=False)].copy()

    exp_c = exp_primary.loc[exp_primary["country_name"].str.lower() == country.lower(), ["year", "value"]].rename(columns={"value": "exp"})
    enr_c = enrol_primary.loc[enrol_primary["country_name"].str.lower() == country.lower(), ["year", "value"]].rename(columns={"value": "enrol"})

    df = pd.merge(exp_c, enr_c, on="year", how="inner").dropna()
    if df.empty:
        raise ValueError(f"No overlapping exp_primary and enrol_primary data for {country}.")

    df = df.sort_values("year")
    df["ratio"] = df["exp"] / df["enrol"]

    fig = plt.figure()
    plt.plot(df["year"].astype(int), df["ratio"])
    plt.title(f"{country}: Ratio (primary expenditure) / (primary enrolment) over time")
    plt.xlabel("Year")
    plt.ylabel("Expenditure / enrolment")
    return save_plot(fig, f"h_ratio_exp_primary_to_enrol_primary_{normalize_country_name(country)}.png")


# ============================================================
# (i) Plot average adult literacy rate across all countries by year
# ============================================================

def task_i_plot_avg_adult_literacy(df_long: pd.DataFrame) -> pd.DataFrame:
    lit = get_indicator(df_long, ["adult literacy rate", "population 15+"])
    df = lit.dropna(subset=["value", "year"]).copy()

    yearly = df.groupby("year", as_index=False)["value"].mean().rename(columns={"value": "avg_adult_literacy_rate"})
    yearly = yearly.sort_values("year")

    fig = plt.figure()
    plt.plot(yearly["year"].astype(int), yearly["avg_adult_literacy_rate"])
    plt.title("Average adult literacy rate across all countries by year")
    plt.xlabel("Year")
    plt.ylabel("Adult literacy rate (%)")
    save_plot(fig, "i_avg_adult_literacy_rate_by_year.png")

    return yearly


# ============================================================
# (j) Visualisation comparing enrolment in 2018 for three countries
# ============================================================

def task_j_plot_enrolment_2018_three_countries(df_long: pd.DataFrame, countries: tuple[str, str, str]) -> pd.DataFrame:
    enrol = get_indicator(df_long, ["enrolment", "both sexes", "number"]).copy()
    enrol["indicator_lc"] = enrol["indicator"].astype(str).str.lower()

    # keep only 2018 and the three education levels (pre-primary/primary/secondary)
    df_2018 = enrol.loc[enrol["year"] == 2018].copy()

    def level(s: str) -> str:
        if "pre-primary" in s or "pre primary" in s:
            return "pre_primary"
        if "secondary" in s:
            return "secondary"
        if "primary" in s and "pre" not in s:
            return "primary"
        return "other"

    df_2018["level"] = df_2018["indicator_lc"].map(level)
    df_2018 = df_2018.loc[df_2018["level"].isin(["pre_primary", "primary", "secondary"])]

    # Filter countries by normalized names (more robust than exact label matching)
    wanted_norm = [normalize_country_name(c) for c in countries]
    df_2018 = df_2018.loc[df_2018["country_norm"].isin(wanted_norm)].copy()

    if df_2018.empty:
        raise ValueError("No 2018 enrolment rows found for the selected countries/levels.")

    # Aggregate in case multiple rows exist
    snap = (
        df_2018.groupby(["country_name", "country_norm", "level"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "enrolment_2018"})
    )

    # Pivot for plotting
    pivot = snap.pivot_table(index="level", columns="country_name", values="enrolment_2018", aggfunc="mean")
    pivot = pivot.reindex(["pre_primary", "primary", "secondary"])

    # Plot: grouped bars
    fig = plt.figure()
    ax = plt.gca()

    levels = pivot.index.tolist()
    country_cols = pivot.columns.tolist()
    x = np.arange(len(levels))
    width = 0.25 if len(country_cols) == 3 else 0.3

    for i, cname in enumerate(country_cols):
        y = pivot[cname].values
        # shift bars to form groups
        ax.bar(x + (i - (len(country_cols) - 1) / 2) * width, y, width=width, label=cname)

    ax.set_title("Enrolment in 2018 by education level (selected countries)")
    ax.set_xlabel("Education level")
    ax.set_ylabel("Enrolment (both sexes, number)")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()

    save_plot(fig, f"j_enrolment_comparison_2018_{'_'.join([normalize_country_name(c) for c in countries])}.png")

    return snap.sort_values(["country_name", "level"])


# ============================================================
# (k) Population > 10 million, sort + avg area
# ============================================================

def task_k_population_over_10m(df_country: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = df_country.copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    big = df.loc[df["population"] > 10_000_000].dropna(subset=["population", "area"]).copy()
    by_pop_asc = big.sort_values("population", ascending=True)
    by_area_desc = big.sort_values("area", ascending=False)
    avg_area = float(big["area"].mean()) if not big.empty else float("nan")
    return by_pop_asc, by_area_desc, avg_area


# ============================================================
# (l) Per region: density + class + top 5
# ============================================================

def task_l_density_by_region_top5(df_country: pd.DataFrame) -> pd.DataFrame:
    df = df_country.copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["population", "area", "region"])
    df = df.loc[df["area"] > 0].copy()

    df["density"] = df["population"] / df["area"]
    df["density_class"] = df["density"].map(classify_density)

    # Top 5 per region by density
    df = df.sort_values(["region", "density"], ascending=[True, False])
    top5 = df.groupby("region", as_index=False).head(5)
    return top5[["region", "country", "population", "area", "density", "density_class"]]


# ============================================================
# (m) Estimate adult illiterates in 2019 (assume adults 65-80% of population)
# ============================================================

def task_m_estimate_adult_illiterates_2019(df_country: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    df_pop = df_country[["country", "country_norm", "population"]].copy()
    df_pop["population"] = pd.to_numeric(df_pop["population"], errors="coerce")
    df_pop = df_pop.dropna(subset=["population"])

    lit = get_indicator(df_long, ["adult literacy rate", "population 15+"]).copy()
    lit = lit.dropna(subset=["value", "year"])
    lit["value"] = pd.to_numeric(lit["value"], errors="coerce")

    # Use literacy for 2019 if available; otherwise use latest <= 2019 per country
    lit = lit.loc[lit["year"] <= 2019].copy()
    lit = lit.sort_values(["country_norm", "year"])
    lit_latest = lit.groupby("country_norm", as_index=False).tail(1)
    lit_latest = lit_latest[["country_norm", "country_name", "year", "value"]].rename(
        columns={"value": "adult_literacy_rate", "year": "literacy_year"}
    )

    merged = pd.merge(df_pop, lit_latest, on="country_norm", how="left")

    # Adult share bounds
    low_share = 0.65
    high_share = 0.80

    # Illiterate share = 1 - literacy_rate
    merged["illiterate_share"] = 1 - (merged["adult_literacy_rate"] / 100.0)

    merged["adult_illiterates_low"] = merged["population"] * low_share * merged["illiterate_share"]
    merged["adult_illiterates_high"] = merged["population"] * high_share * merged["illiterate_share"]

    return merged[["country", "population", "adult_literacy_rate", "literacy_year", "adult_illiterates_low", "adult_illiterates_high"]]


# ============================================================
# (n) For each region: average secondary enrolment in 2015. Which region leads?
# ============================================================

def task_n_avg_secondary_enrolment_2015_by_region(df_country: pd.DataFrame, df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    enrol_secondary = get_indicator(df_long, ["enrolment", "secondary", "both sexes", "number"])
    df_2015 = enrol_secondary.loc[enrol_secondary["year"] == 2015].dropna(subset=["value"]).copy()

    # join region via df_country country_norm
    region_map = df_country[["country_norm", "region"]].dropna()
    merged = pd.merge(df_2015, region_map, on="country_norm", how="left").dropna(subset=["region"])

    region_avg = merged.groupby("region", as_index=False)["value"].mean().rename(columns={"value": "avg_secondary_enrolment_2015"})
    region_avg = region_avg.sort_values("avg_secondary_enrolment_2015", ascending=False)

    top_region = region_avg.iloc[0] if not region_avg.empty else pd.Series(dtype=object)
    return region_avg, top_region


# ============================================================
# (o) Ten smallest countries by area: primary enrolment in 2019
# ============================================================

def task_o_primary_enrolment_2019_smallest_area(df_country: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_country.copy()
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    smallest10 = df.dropna(subset=["area"]).sort_values("area", ascending=True).head(10)[["country", "country_norm", "area"]]

    enrol_primary = get_indicator(df_long, ["enrolment", "primary", "both sexes", "number"])
    enrol_primary = enrol_primary.loc[~enrol_primary["indicator"].astype(str).str.lower().str.contains("pre-primary", na=False)].copy()

    df_2019 = enrol_primary.loc[enrol_primary["year"] == 2019].copy()
    df_2019 = df_2019.groupby("country_norm", as_index=False)["value"].mean().rename(columns={"value": "primary_enrolment_2019"})

    out = pd.merge(smallest10, df_2019, on="country_norm", how="left")
    return out.sort_values("area", ascending=True)


# ============================================================
# (p) Correlation between area and primary enrolment in 2015 (+ optional plot)
# ============================================================

def task_p_area_vs_primary_enrolment_corr_2015(df_country: pd.DataFrame, df_long: pd.DataFrame) -> tuple[float, pd.DataFrame, Path]:
    df = df_country.copy()
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    area_df = df[["country", "country_norm", "area"]].dropna(subset=["area"])

    enrol_primary = get_indicator(df_long, ["enrolment", "primary", "both sexes", "number"])
    enrol_primary = enrol_primary.loc[~enrol_primary["indicator"].astype(str).str.lower().str.contains("pre-primary", na=False)].copy()

    df_2015 = enrol_primary.loc[enrol_primary["year"] == 2015].copy()
    enrol_df = df_2015.groupby("country_norm", as_index=False)["value"].mean().rename(columns={"value": "primary_enrolment_2015"})

    merged = pd.merge(area_df, enrol_df, on="country_norm", how="inner").dropna()

    if merged.empty:
        raise ValueError("No overlapping area + primary enrolment (2015) data to compute correlation.")

    corr = float(merged["area"].corr(merged["primary_enrolment_2015"], method="pearson"))

    fig = plt.figure()
    plt.scatter(merged["area"], merged["primary_enrolment_2015"])
    plt.title("Area vs Primary enrolment (2015)")
    plt.xlabel("Area (square kilometres)")
    plt.ylabel("Primary enrolment (2015)")
    plot_path = save_plot(fig, "p_area_vs_primary_enrolment_2015_scatter.png")

    return corr, merged.sort_values("area"), plot_path


# ============================================================
# Main runner (a-p)
# ============================================================

def main() -> None:
    ensure_dirs()

    # (a)
    df_country, df_education, in_country_not_edu, in_edu_not_country = task_a_load_and_compare()
    print("(a) Loaded:")
    print("df_country shape:", df_country.shape)
    print("df_education shape:", df_education.shape)

    print("\n(a) Country names present in countries_data but not education (normalised) - sample:")
    print(in_country_not_edu[:20])

    print("\n(a) Country names present in education but not countries_data (normalised) - sample:")
    print(in_edu_not_country[:20])

    # (b)
    df_education_filtered = task_b_filter_common(df_country, df_education)
    print("\n(b) df_education_filtered shape:", df_education_filtered.shape)
    print("(b) Sample rows:")
    print(df_education_filtered[["Country Name", "Country Code", "Series"]].head(5))

    # (c)
    df_education_clean = task_c_clean_transform(df_education_filtered)
    print("\n(c) df_education_clean (long format) shape:", df_education_clean.shape)
    print(df_education_clean.head(5))

    # (d)
    max_sec_2019 = task_d_max_secondary_teachers_2019(df_education_clean)
    print("\n(d) Country with max secondary teachers in 2019:")
    print(max_sec_2019)

    # (e)
    print(f"\n(e) Plotting secondary expenditure trend for {COUNTRY_E} ...")
    e_path = task_e_plot_secondary_expenditure(df_education_clean, COUNTRY_E)
    print("Saved to:", e_path)

    # (f)
    print("\n(f) Computing teachers-to-enrolment ratios ...")
    ratio_df = task_f_teacher_to_enrolment_ratio(df_education_clean)
    print("Ratio table shape:", ratio_df.shape)
    show_f = task_f_show_two_countries(ratio_df, COUNTRIES_F)
    print(f"\n(f) Sample ratios for selected countries {COUNTRIES_F}:")
    print(show_f.head(20))

    # (g)
    max_inc, max_dec = task_g_primary_enrolment_change_2019(df_education_clean)
    print("\n(g) Primary enrolment change (2018 -> 2019):")
    print("Largest increase:")
    print(max_inc)
    print("Largest decrease:")
    print(max_dec)

    # (h)
    print(f"\n(h) Plotting ratio (exp primary)/(enrol primary) for {COUNTRY_H} ...")
    h_path = task_h_plot_exp_to_primary_enrol_ratio(df_education_clean, COUNTRY_H)
    print("Saved to:", h_path)

    # (i)
    yearly_lit = task_i_plot_avg_adult_literacy(df_education_clean)
    print("\n(i) Adult literacy yearly mean head:")
    print(yearly_lit.head(10))

    # (j)
    print(f"\n(j) Plotting enrolment 2018 comparison for {COUNTRIES_J} ...")
    snap_2018 = task_j_plot_enrolment_2018_three_countries(df_education_clean, COUNTRIES_J)
    print("(j) 2018 enrolment snapshot table:")
    print(snap_2018)

    # (k)
    by_pop_asc, by_area_desc, avg_area = task_k_population_over_10m(df_country)
    print("\n(k) Countries with population > 10M:")
    print("Sorted by population ASC (head 10):")
    print(by_pop_asc[["country", "population", "area"]].head(10))
    print("Sorted by area DESC (head 10):")
    print(by_area_desc[["country", "population", "area"]].head(10))
    print("Average area (pop > 10M):", avg_area)

    # (l)
    top5_density = task_l_density_by_region_top5(df_country)
    print("\n(l) Top 5 countries by density in each region:")
    print(top5_density)

    # (m)
    illit = task_m_estimate_adult_illiterates_2019(df_country, df_education_clean)
    print("\n(m) Estimated adult illiterates in 2019 (bounds using adult share 65%-80%) - head 15:")
    print(illit.head(15))

    # (n)
    region_avg, top_region = task_n_avg_secondary_enrolment_2015_by_region(df_country, df_education_clean)
    print("\n(n) Average secondary enrolment in 2015 by region (top 10):")
    print(region_avg.head(10))
    print("\n(n) Region that leads:")
    print(top_region)

    # (o)
    smallest10_enrol = task_o_primary_enrolment_2019_smallest_area(df_country, df_education_clean)
    print("\n(o) Primary enrolment (2019) for 10 smallest countries by area:")
    print(smallest10_enrol)

    # (p)
    corr, p_table, p_plot = task_p_area_vs_primary_enrolment_corr_2015(df_country, df_education_clean)
    print("\n(p) Correlation (Pearson) between area and primary enrolment in 2015:", corr)
    print("Scatter saved to:", p_plot)

    print("\nDone ✅  Check outputs/ for saved plots.")


if __name__ == "__main__":
    main()