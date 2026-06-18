"""
Live data loader for NHS A&E Type 1 monthly performance data.

Primary path: fetch the most recent 13 contiguous months directly from
NHS England's published CSVs (current + previous financial year index pages).
Caching is handled by the caller (Streamlit's @st.cache_data).

Fallback: the caller supplies a path to recent_history.csv when the live
fetch fails or yields fewer than the requested number of months.
"""
import re
from io import BytesIO

import pandas as pd
import requests

BASE = "https://www.england.nhs.uk"
HEADERS = {"User-Agent": "ae-panel-research/1.0 (open-data)"}

YEAR_PAGES = {
    "2026-27": BASE + "/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2026-27/",
    "2025-26": BASE + "/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2025-26/",
    "2024-25": BASE + "/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2024-25/",
    "2023-24": BASE + "/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2023-24/",
    "2022-23": BASE + "/statistics/statistical-work-areas/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2022-23/",
}

MONTHS = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}

OUTPUT_COLS = ["org_code", "org_name", "month", "within_4hrs", "attendances", "admissions"]


def find_csv_links(page_html: str) -> list:
    """Extract monthly CSV links from an NHS England financial-year index page."""
    links = re.findall(r'href="([^"]+\.csv)"', page_html, flags=re.IGNORECASE)
    monthly = [lnk for lnk in links if "quarter" not in lnk.lower()]
    full = [lnk if lnk.startswith("http") else BASE + lnk for lnk in monthly]
    return list(dict.fromkeys(full))


def load_month(path) -> pd.DataFrame:
    """
    Parse one monthly A&E CSV (file path or file-like object) into a tidy
    per-hospital Type 1 performance table.

    Columns returned: org_code, org_name, month, within_4hrs, attendances, admissions.
    Rows with no org_code, zero Type 1 attendances, or 'TOTAL' summary lines are dropped.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().replace("number of ", "").strip() for c in df.columns]

    period = str(df["period"].iloc[0]).upper().split("-")
    month = pd.Timestamp(int(period[-1]), MONTHS[period[-2]], 1)

    out = pd.DataFrame({
        "month": month,
        "org_code": df["org code"].str.strip(),
        "org_name": df["org name"].str.strip(),
        "attendances": pd.to_numeric(df["a&e attendances type 1"], errors="coerce"),
        "over_4hrs": pd.to_numeric(df["attendances over 4hrs type 1"], errors="coerce"),
        "admissions": pd.to_numeric(df["emergency admissions via a&e - type 1"], errors="coerce"),
    })

    out = out.dropna(subset=["org_code"])
    out = out[out["org_code"].str.upper() != "TOTAL"]
    out = out[out["attendances"] > 0].copy()
    out["within_4hrs"] = 1.0 - out["over_4hrs"] / out["attendances"]

    return out[OUTPUT_COLS].reset_index(drop=True)


def _fy_keys_to_check() -> list:
    """Current and two preceding financial year keys, most recent first."""
    today = pd.Timestamp.today()
    fy_start = today.year if today.month >= 4 else today.year - 1
    return [
        f"{fy_start}-{str(fy_start + 1)[-2:]}",
        f"{fy_start - 1}-{str(fy_start)[-2:]}",
        f"{fy_start - 2}-{str(fy_start - 1)[-2:]}",
    ]


def fetch_live_data(n_months: int = 13, timeout: int = 20) -> pd.DataFrame:
    """
    Fetch the most recent n_months of Type 1 A&E data from NHS England.

    Reads financial-year index pages (current then previous) and downloads
    every monthly CSV found, stopping once n_months of unique calendar months
    have been collected. Returns a DataFrame with OUTPUT_COLS, sorted by
    (org_code, month).

    Raises RuntimeError if the live fetch yields fewer than n_months distinct
    calendar months — the caller should then fall back to the bundled snapshot.
    """
    accumulated: list = []

    for year_key in _fy_keys_to_check():
        page_url = YEAR_PAGES.get(year_key)
        if not page_url:
            continue

        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            links = find_csv_links(resp.text)
        except Exception as exc:
            raise RuntimeError(f"Could not fetch index page {year_key}: {exc}") from exc

        for url in links:
            try:
                content = requests.get(url, headers=HEADERS, timeout=timeout).content
                month_df = load_month(BytesIO(content))
                if len(month_df) > 0:
                    accumulated.append(month_df)
            except Exception:
                continue

        if accumulated:
            unique_so_far = pd.concat(accumulated, ignore_index=True)["month"].nunique()
            if unique_so_far >= n_months:
                break

    if not accumulated:
        raise RuntimeError("Live fetch returned no data.")

    combined = (
        pd.concat(accumulated, ignore_index=True)
        .drop_duplicates(["org_code", "month"])
        .sort_values(["org_code", "month"])
        .reset_index(drop=True)
    )

    recent_months = sorted(combined["month"].unique())[-n_months:]
    combined = combined[combined["month"].isin(recent_months)].reset_index(drop=True)

    if combined["month"].nunique() < n_months:
        raise RuntimeError(
            f"Live fetch returned only {combined['month'].nunique()} months; need {n_months}."
        )

    return combined
