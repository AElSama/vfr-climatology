"""Aerodromes and METAR archive download (Iowa Environmental Mesonet)."""
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Home field: Saint-Cyr-l'Ecole has no METAR, so Toussus-le-Noble (8 km south) is the reference station.
HOME = {"icao": "LFPZ", "name": "Saint-Cyr-l'École", "lat": 48.8114, "lon": 2.0747, "runway": (110, 290)}
REFERENCE = "LFPN"

# METAR stations around Paris (approximate aerodrome reference points)
STATIONS = {
    "LFPN": ("Toussus-le-Noble", 48.7519, 2.1061),
    "LFPV": ("Vélizy-Villacoublay", 48.7742, 2.2017),
    "LFPO": ("Paris-Orly", 48.7233, 2.3794),
    "LFPG": ("Paris-Charles-de-Gaulle", 49.0097, 2.5479),
    "LFPB": ("Paris-Le Bourget", 48.9694, 2.4414),
    "LFPT": ("Pontoise-Cormeilles", 49.0967, 2.0408),
    "LFPM": ("Melun-Villaroche", 48.6047, 2.6711),
    "LFOB": ("Beauvais-Tillé", 49.4544, 2.1128),
    "LFOE": ("Évreux-Fauville", 49.0287, 1.2199),
    "LFOJ": ("Orléans-Bricy", 47.9878, 1.7606),
}


def download_year(icao: str, year: int, retries: int = 3) -> pd.DataFrame:
    """Raw METAR/SPECI of one station for one calendar year (UTC timestamps)."""
    query = urllib.parse.urlencode({
        "station": icao, "data": "metar", "tz": "Etc/UTC", "format": "onlycomma",
        "latlon": "no", "missing": "M", "trace": "T", "direct": "no",
        "year1": year, "month1": 1, "day1": 1, "year2": year + 1, "month2": 1, "day2": 1,
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{IEM_URL}?{query}", timeout=300) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text), comment="#")
            return df[["station", "valid", "metar"]]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def load_station(icao: str, years: range, cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download (once) and cache a station's METAR archive. Returns columns [time_utc, metar]."""
    path = Path(cache_dir) / f"{icao}_{years.start}_{years.stop - 1}.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
    else:
        parts = []
        for y in years:
            parts.append(download_year(icao, y))
            time.sleep(1.0)  # be gentle with the free service
        df = pd.concat(parts, ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, compression="gzip")
    df = df[df["metar"].astype(str).str.len() > 10].copy()
    df["time_utc"] = pd.to_datetime(df["valid"], utc=True)
    return df.drop_duplicates("time_utc").sort_values("time_utc")[["time_utc", "metar"]].reset_index(drop=True)


def synthetic_station(icao: str = "LFPN", start: str = "2015-01-01", end: str = "2024-12-31",
                      seed: int = 0) -> pd.DataFrame:
    """Half-hourly METAR strings with realistic seasonality (fog in autumn mornings,
    low cloud in winter, convection in summer afternoons), to test the pipeline offline."""
    rng = np.random.default_rng(seed + sum(map(ord, icao)))
    t = pd.date_range(start, end + " 23:30", freq="30min", tz="UTC")
    n = len(t)
    doy, hour = t.dayofyear.to_numpy(), t.hour.to_numpy() + t.minute.to_numpy() / 60
    winter = 0.5 * (1 + np.cos(2 * np.pi * (doy - 15) / 365))           # 1 in January, 0 in July
    autumn = np.exp(-((doy - 300) / 45.0) ** 2)
    morning = np.exp(-((hour - 6.5) / 2.0) ** 2)
    afternoon = np.exp(-((hour - 15.0) / 2.5) ** 2)

    # slow weather regimes (AR(1) on a 30-min grid ~ 1.5 day memory)
    def ar1(phi, size):
        x = np.zeros(size)
        e = rng.standard_normal(size)
        for i in range(1, size):
            x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * e[i]
        return x

    reg_cloud, reg_wind = ar1(0.985, n), ar1(0.98, n)
    p_fog = 1 / (1 + np.exp(-(-4.2 + 2.5 * autumn + 1.8 * winter + 2.5 * morning + 0.8 * reg_cloud)))
    p_low = 1 / (1 + np.exp(-(-2.0 + 2.2 * winter + 1.5 * reg_cloud)))
    p_ts = 1 / (1 + np.exp(-(-7.0 + 4.0 * (1 - winter) * afternoon + 0.5 * reg_cloud)))
    wind = np.clip(8 + 4 * winter + 5 * reg_wind + 2 * rng.standard_normal(n), 0, 40)
    temp = 12 - 7 * np.cos(2 * np.pi * (doy - 20) / 365) + 4 * np.sin(np.pi * (hour - 9) / 12)
    out = []
    for i in range(n):
        u = rng.random(3)
        ddd = int(rng.choice([240, 250, 260, 220, 40, 60, 300, 180]))
        ff = int(round(wind[i]))
        g = f"G{ff + int(rng.integers(8, 15))}" if ff >= 15 and u[2] < 0.6 else ""
        w = f"{ddd:03d}{ff:02d}{g}KT" if ff > 0 else "00000KT"
        tt, td = temp[i], temp[i] - (1 if u[0] < p_fog[i] else rng.uniform(2, 9))
        if u[0] < p_fog[i]:
            body = f"{int(rng.choice([150, 300, 600, 900])):04d} FG VV00{int(rng.integers(1, 3))}"
        elif u[2] < p_ts[i]:
            body = f"5000 TSRA SCT030CB BKN0{int(rng.integers(45, 80))}"
        elif u[1] < p_low[i]:
            body = f"{int(rng.choice([4000, 6000, 8000, 9999])):04d} BKN0{int(rng.integers(6, 30)):02d}"
        else:
            body = "CAVOK" if u[1] > 0.7 else f"9999 SCT0{int(rng.integers(25, 50))}"
        ts = t[i].strftime("%d%H%MZ")
        out.append(f"{icao} {ts} {w} {body} {_fmt(tt)}/{_fmt(td)} Q1015")
    return pd.DataFrame({"time_utc": t, "metar": out})


def _fmt(x: float) -> str:
    v = int(round(x))
    return f"M{abs(v):02d}" if v < 0 else f"{v:02d}"
