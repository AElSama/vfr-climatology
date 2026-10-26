import numpy as np
import pandas as pd

from src.data import synthetic_station
from src.flying import (STUDENT, crosswind, evaluate, hourly_conditions, slot_table, sun_elevation)


def test_sun_elevation_paris():
    t = pd.DatetimeIndex(["2024-06-21 11:50", "2024-12-21 11:50", "2024-06-21 23:00"], tz="UTC")
    e = sun_elevation(t, 48.85, 2.35)
    assert abs(e[0] - 64.6) < 1.0 and abs(e[1] - 17.7) < 1.0 and e[2] < 0


def test_crosswind_on_runway_11_29():
    xw = crosswind([110, 200, 290, np.nan], [20, 20, 10, 8], (110, 290))
    assert np.allclose(xw, [0, 20, 0, 8], atol=1e-9)


def _hourly(metars, start="2024-06-15 08:00"):
    t = pd.date_range(start, periods=len(metars), freq="h", tz="UTC")
    return hourly_conditions(pd.DataFrame({"time_utc": t, "metar": metars}), 48.75, 2.11)


def test_reasons_are_flagged():
    h = _hourly(["LFPN 150800Z 11008KT CAVOK 20/10 Q1020",
                 "LFPN 150900Z 20018KT 9999 SCT040 20/10 Q1020",
                 "LFPN 151000Z 11005KT 9999 BKN012 20/10 Q1020",
                 "LFPN 151100Z 11005KT 0300 FG VV001 12/12 Q1020"])
    ev = evaluate(h, STUDENT)
    assert ev["flyable"].tolist() == [True, False, False, False]
    assert ev.loc[1, "crosswind"] and ev.loc[1, "wind"]
    assert ev.loc[2, "ceiling"] and not ev.loc[2, "visibility"]
    assert ev.loc[3, "visibility"] and ev.loc[3, "weather"]


def test_two_hour_slot_needs_three_good_observations():
    good, bad = "LFPN 150800Z 11005KT CAVOK 20/10 Q1020", "LFPN 150800Z 11005KT 9999 OVC008 20/10 Q1020"
    h = _hourly([good, good, good, bad, good])
    ev = evaluate(h, STUDENT)
    tab = slot_table(ev, 2)
    # start 08 UTC (10h local): obs 08, 09, 10 good -> flyable; start 09 UTC: 09, 10, 11(bad) -> not
    assert tab.loc[6, 10] == 1.0 and tab.loc[6, 11] == 0.0


def test_pipeline_on_synthetic_data_has_a_seasonal_cycle():
    m = synthetic_station("LFPN", "2023-01-01", "2023-12-31")
    ev = evaluate(hourly_conditions(m, 48.75, 2.11), STUDENT)
    d = ev[ev["daylight"] & ev["observed"]].groupby("month")["flyable"].mean()
    assert d.loc[[6, 7]].mean() > d.loc[[11, 12]].mean() + 0.15
