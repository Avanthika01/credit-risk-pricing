"""Geographic fair-lending proxy review (adverse impact ratio, four-fifths rule).

IMPORTANT LIMITATION: Lending Club data contains NO protected attributes (race,
ethnicity, sex, age, etc.). This is therefore a geographic, redlining-style proxy
review by state and 3-digit ZIP - NOT a full fair-lending analysis, which would
require protected-class data or proxy methods such as BISG.

Also note every loan in the data was already approved by Lending Club, so the
"approval rate" here is the share our model WOULD approve at the cutoff among
booked loans, not a rate over the true applicant population.
"""

from __future__ import annotations

import pandas as pd

LIMITATION_NOTE = (
    "Lending Club data has no protected attributes. This is a geographic (redlining-style) "
    "proxy review by state and 3-digit ZIP, not a full fair-lending analysis. Approval rates "
    "are hypothetical model decisions over loans Lending Club already booked."
)


def zip3(zip_code: pd.Series) -> pd.Series:
    return zip_code.astype("string").str[:3]


def adverse_impact(df: pd.DataFrame, group_col: str, pd_col: str = "pd", target_col: str = "default",
                   cutoff: float = 0.20, min_group_size: int = 500, air_threshold: float = 0.80) -> pd.DataFrame:
    """Approval rate and adverse impact ratio by group.

    approve = PD <= cutoff. AIR = group approval rate / highest group approval rate
    (among groups with at least `min_group_size` loans). AIR < 0.80 is flagged.
    """
    d = df.assign(_approved=(df[pd_col] <= cutoff).astype(float))
    g = d.groupby(group_col, observed=True).agg(
        n=("_approved", "size"),
        approval_rate=("_approved", "mean"),
        mean_pd=(pd_col, "mean"),
        observed_default_rate=(target_col, "mean"),
    )
    g = g.loc[g["n"] >= min_group_size].copy()
    if g.empty:
        return g.assign(air=pd.Series(dtype=float), flagged=pd.Series(dtype=bool))
    g["air"] = g["approval_rate"] / g["approval_rate"].max()
    g["flagged"] = g["air"] < air_threshold
    return g.sort_values("air").reset_index()


def fairness_summary(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"n_groups": 0, "n_flagged": 0}
    return {
        "n_groups": int(len(table)),
        "n_flagged": int(table["flagged"].sum()),
        "min_air": float(table["air"].min()),
        "min_air_group": str(table.iloc[0, 0]),
        "flagged_groups": [str(x) for x in table.loc[table["flagged"]].iloc[:, 0]],
    }
