"""Flyability from METARs: daylight, personal minimums, hourly grid, booking slots, morning forecast."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metar import decode_frame

LOCAL_TZ = "Europe/Paris"


@dataclass(frozen=True)
class Minima:
    name: str
    vis_m: float
    ceiling_ft: float
    wind_kt: float = np.inf
    gust_kt: float = np.inf
    crosswind_kt: float = np.inf
    block_wx: tuple = ("TS", "FG", "FZ", "SN", "GR", "+")


# A student pilot's personal minimums (conservative, adjust to your instructor's rules)
STUDENT = Minima("Student minimums", vis_m=8000, ceiling_ft=2000, wind_kt=15, gust_kt=20, crosswind_kt=10)
# VFR in a control zone (SERA): 5 km visibility, 1,500 ft ceiling
VFR = Minima("Regulatory VFR (CTR)", vis_m=5000, ceiling_ft=1500, block_wx=("TS", "FG"))


def sun_elevation(times_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Solar elevation in degrees (NOAA approximation, accurate to a fraction of a degree)."""
    t = pd.DatetimeIndex(times_utc)
    doy = t.dayofyear.to_numpy()
    hours = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hours * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cosz, -1, 1)))


def crosswind(wind_dir, wind_kt, runway_headings) -> np.ndarray:
    """Crosswind on the best runway. Variable wind (VRB) counts as full crosswind (conservative)."""
    wind_dir, wind_kt = np.asarray(wind_dir, float), np.asarray(wind_kt, float)
    xw = np.full(wind_kt.shape, np.inf)
    for h in runway_headings:
        xw = np.minimum(xw, np.abs(wind_kt * np.sin(np.radians(wind_dir - h))))
    return np.where(np.isnan(wind_dir), wind_kt, xw)


def hourly_conditions(metars: pd.DataFrame, lat: float, lon: float, runway=(110, 290),
                      max_age_min: int = 90) -> pd.DataFrame:
    """One row per UTC hour: the latest METAR at or before the hour (if not older than max_age_min)."""
    dec = decode_frame(metars["metar"])
    dec["time_utc"] = metars["time_utc"].to_numpy()
    grid = pd.date_range(dec["time_utc"].min().ceil("h"), dec["time_utc"].max().floor("h"), freq="h", tz="UTC")
    hourly = pd.merge_asof(pd.DataFrame({"time_utc": grid}), dec.sort_values("time_utc"), on="time_utc",
                           direction="backward", tolerance=pd.Timedelta(minutes=max_age_min))
    gust = hourly["gust_kt"].fillna(hourly["wind_kt"])
    hourly["crosswind_kt"] = crosswind(hourly["wind_dir"], gust, runway)
    hourly["sun_elev"] = sun_elevation(pd.DatetimeIndex(hourly["time_utc"]), lat, lon)
    local = pd.DatetimeIndex(hourly["time_utc"]).tz_convert(LOCAL_TZ)
    hourly["local"] = local
    hourly["date"] = local.date
    hourly["hour"] = local.hour
    hourly["month"] = local.month
    hourly["year"] = local.year
    hourly["observed"] = hourly["vis_m"].notna()
    return hourly


def evaluate(hourly: pd.DataFrame, minima: Minima) -> pd.DataFrame:
    """Reason flags and flyable = daylight, observed, and every criterion met."""
    h = hourly
    wx = h["wx"].fillna("")
    flags = pd.DataFrame({
        "visibility": h["vis_m"] < minima.vis_m,
        "ceiling": h["ceiling_ft"] < minima.ceiling_ft,
        "wind": (h["wind_kt"] > minima.wind_kt) | (h["gust_kt"].fillna(0) > minima.gust_kt),
        "crosswind": h["crosswind_kt"] > minima.crosswind_kt,
        "weather": wx.apply(lambda s: any(c in s for c in minima.block_wx)) | (h["cb"].fillna(False).astype(bool)
                                                                           & ("TS" in minima.block_wx)),
    }, index=h.index)
    out = h[["time_utc", "local", "date", "hour", "month", "year", "observed"]].copy()
    out["daylight"] = h["sun_elev"] > 0
    out = pd.concat([out, flags], axis=1)
    out["flyable"] = out["observed"] & ~flags.any(axis=1)
    return out


def month_hour_table(ev: pd.DataFrame) -> pd.DataFrame:
    """P(flyable) by month x local hour, over observed daylight hours (NaN where < 50% daylight)."""
    d = ev[ev["observed"]]
    p = d[d["daylight"]].pivot_table(index="month", columns="hour", values="flyable", aggfunc="mean")
    day_share = d.pivot_table(index="month", columns="hour", values="daylight", aggfunc="mean")
    return p.where(day_share.reindex_like(p) >= 0.5)


def reasons_by_month(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev[ev["observed"] & ev["daylight"]]
    cols = ["visibility", "ceiling", "wind", "crosswind", "weather"]
    return d.groupby("month")[cols + ["flyable"]].mean()


def slot_table(ev: pd.DataFrame, length_h: int = 2) -> pd.DataFrame:
    """P(a lesson of `length_h` hours starting at hour h is flyable throughout), by month x start hour.

    Requires the METAR at h, h+1, ..., h+length_h to be flyable and in daylight.
    """
    e = ev.set_index("time_utc").sort_index()
    ok = (e["flyable"] & e["daylight"]).astype(float).where(e["observed"])
    window = ok.copy()
    for k in range(1, length_h + 1):
        window = window * ok.shift(-k)
    ends_in_daylight = e["daylight"].shift(-length_h, fill_value=False).astype(bool)
    e = e.assign(slot=window.where(ends_in_daylight))  # a lesson must end before sunset
    d = e[e["daylight"] & e["slot"].notna()]
    return d.pivot_table(index="month", columns="hour", values="slot", aggfunc="mean")


def day_has_slot(ev: pd.DataFrame, length_h: int = 2) -> pd.DataFrame:
    """Per day: is there at least one flyable lesson slot? With day of week, for 'can I fly on Saturday?'."""
    e = ev.set_index("time_utc").sort_index()
    ok = (e["flyable"] & e["daylight"]).astype(float).where(e["observed"])
    window = ok.copy()
    for k in range(1, length_h + 1):
        window = window * ok.shift(-k)
    e = e.assign(slot=window)
    days = e[e["daylight"]].groupby("date").agg(any_slot=("slot", "max"), month=("month", "first"))
    days["weekday"] = pd.to_datetime(pd.Series(days.index, index=days.index)).dt.dayofweek
    return days.dropna()


def morning_dataset(hourly: pd.DataFrame, ev: pd.DataFrame, morning_hour: int = 7,
                    target_start: int = 14, length_h: int = 2) -> pd.DataFrame:
    """One row per day: features from the METAR at `morning_hour` local, target = afternoon slot flyable."""
    h = hourly.assign(flyable=ev["flyable"], daylight=ev["daylight"], observed=ev["observed"])
    m = h[(h["hour"] == morning_hour) & h["observed"]].set_index("date")
    rows = h[(h["hour"] >= target_start) & (h["hour"] <= target_start + length_h)]
    tgt = rows.groupby("date").agg(n=("observed", "sum"), ok=("flyable", "all"), day=("daylight", "all"))
    tgt = tgt[(tgt["n"] == length_h + 1) & tgt["day"]]
    df = m.join(tgt["ok"], how="inner")
    ceil = np.minimum(df["ceiling_ft"].replace(np.inf, 10000.0), 10000.0)
    feats = pd.DataFrame({
        "log_vis": np.log(df["vis_m"].clip(lower=50)),
        "log_ceiling": np.log(ceil.clip(lower=100)),
        "wind_kt": df["wind_kt"].fillna(0),
        "gust_kt": df["gust_kt"].fillna(df["wind_kt"]).fillna(0),
        "spread_c": (df["temp_c"] - df["dew_c"]).clip(lower=0, upper=20),
        "fog": df["wx"].fillna("").str.contains("FG|BR").astype(float),
        "precip": df["wx"].fillna("").str.contains("RA|DZ|SN|TS").astype(float),
        "morning_flyable": df["flyable"].astype(float),
        "m_cos": np.cos(2 * np.pi * df["month"] / 12),
        "m_sin": np.sin(2 * np.pi * df["month"] / 12),
    }, index=df.index)
    feats["month"] = df["month"].to_numpy()
    feats["year"] = df["year"].to_numpy()
    feats["target"] = df["ok"].astype(float).to_numpy()
    return feats.dropna()


FEATURES = ["log_vis", "log_ceiling", "wind_kt", "gust_kt", "spread_c", "fog", "precip",
            "morning_flyable", "m_cos", "m_sin"]


def forecast_skill(data: pd.DataFrame, test_from_year: int) -> dict:
    """Logistic regression trained before `test_from_year`, scored after it, against climatology."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = data[data["year"] < test_from_year], data[data["year"] >= test_from_year]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(train[FEATURES], train["target"])
    p = clf.predict_proba(test[FEATURES])[:, 1]
    clim = train.groupby("month")["target"].mean()
    p_clim = test["month"].map(clim).to_numpy()
    y = test["target"].to_numpy()
    brier, brier_clim = np.mean((p - y) ** 2), np.mean((p_clim - y) ** 2)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    reliability = [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
                   for b in range(10) if (idx == b).sum() >= 10]
    coefs = dict(zip(FEATURES, clf[-1].coef_[0]))
    return {"n_train": len(train), "n_test": len(test), "brier": float(brier), "brier_climatology": float(brier_clim),
            "brier_skill": float(1 - brier / brier_clim), "auc": float(roc_auc_score(y, p)),
            "reliability": reliability, "coefficients": {k: float(v) for k, v in coefs.items()}, "model": clf,
            "p_test": p, "y_test": y, "test_index": test.index}
