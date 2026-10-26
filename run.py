"""When can a student pilot fly from Saint-Cyr? VFR climatology from 15 years of METARs.

    python run.py              # downloads METAR archives (Iowa Environmental Mesonet) once, then caches
    python run.py --synthetic  # offline run on synthetic METARs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.data import HOME, REFERENCE, STATIONS, load_station, synthetic_station
from src.flying import (STUDENT, VFR, day_has_slot, evaluate, forecast_skill, hourly_conditions,
                        month_hour_table, morning_dataset, reasons_by_month, slot_table)
from src.mapping import build_map

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REF_YEARS = range(2010, 2026)       # reference station: 16 years
MAP_YEARS = range(2018, 2026)       # other stations: 8 years (faster download)
TEST_FROM = 2022                    # forecast: train 2010-2021, test 2022-2025


def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def heatmap(table: pd.DataFrame, title: str, name: str, cbar_label: str):
    hours = list(range(6, 22))
    t = table.reindex(index=range(1, 13), columns=hours)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    im = ax.imshow(t.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=max(0.8, np.nanmax(t.to_numpy(dtype=float))), aspect="auto")
    for i in range(12):
        for j, h in enumerate(hours):
            v = t.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.5 else INK)
    ax.set_xticks(range(len(hours)), [f"{h}h" for h in hours])
    ax.set_yticks(range(12), MONTHS)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    style(ax, title, "Local time", "")
    save(fig, name)


def analyse_station(icao, years, synthetic):
    name, lat, lon = STATIONS[icao]
    raw = synthetic_station(icao, f"{years.start}-01-01", f"{years.stop - 1}-12-31") if synthetic \
        else load_station(icao, years)
    hourly = hourly_conditions(raw, lat, lon, runway=HOME["runway"])
    return hourly, evaluate(hourly, STUDENT)


def main(synthetic: bool):
    Path("figures").mkdir(exist_ok=True)
    hourly, ev = analyse_station(REFERENCE, REF_YEARS, synthetic)
    ev_vfr = evaluate(hourly, VFR)
    obs_day = ev[ev["observed"] & ev["daylight"]]

    table = month_hour_table(ev)
    reasons = reasons_by_month(ev)
    reasons_vfr = reasons_by_month(ev_vfr)
    slots = slot_table(ev, 2)
    days = day_has_slot(ev, 2)
    weekend = days[days["weekday"] >= 5].groupby("month")["any_slot"].mean()

    data = morning_dataset(hourly, ev, morning_hour=7, target_start=14, length_h=2)
    fc = forecast_skill(data, TEST_FROM)
    cond = data.groupby(data["fog"] == 1)["target"].mean()

    # best slots: top 3 (month-specific) start hours by probability
    best = {MONTHS[m - 1]: {"start": f"{int(slots.loc[m].idxmax())}h", "p": round(float(slots.loc[m].max()), 3)}
            for m in slots.index if slots.loc[m].notna().any()}

    # stations for the map
    map_rows = []
    for icao in STATIONS:
        try:
            _, e = (hourly, ev) if icao == REFERENCE else analyse_station(icao, MAP_YEARS, synthetic)
        except Exception as exc:  # a station may be missing from the archive
            print(f"skip {icao}: {exc}")
            continue
        d = e[e["observed"] & e["daylight"]]
        if len(d) < 1000:
            continue
        monthly = d.groupby("month")["flyable"].mean().reindex(range(1, 13)).fillna(0)
        yrs = REF_YEARS if icao == REFERENCE else MAP_YEARS
        map_rows.append({"icao": icao, "name": STATIONS[icao][0], "lat": STATIONS[icao][1], "lon": STATIONS[icao][2],
                         "annual": round(float(d["flyable"].mean()), 4),
                         "monthly": [round(float(x), 4) for x in monthly], "n_hours": int(len(d)),
                         "years": f"{yrs.start}–{yrs.stop - 1}"})

    results = {
        "data": {"source": "synthetic" if synthetic else "Iowa Environmental Mesonet METAR archive",
                 "reference": REFERENCE, "home": HOME["icao"], "years": f"{REF_YEARS.start}-{REF_YEARS.stop - 1}",
                 "hours_observed_daylight": int(len(obs_day)), "metar_coverage": round(float(ev["observed"].mean()), 3)},
        "minima": {"student": STUDENT.__dict__, "vfr": VFR.__dict__},
        "flyable_share": {
            "student_annual": round(float(obs_day["flyable"].mean()), 3),
            "vfr_annual": round(float(ev_vfr[ev_vfr["observed"] & ev_vfr["daylight"]]["flyable"].mean()), 3),
            "student_by_month": {MONTHS[m - 1]: round(float(v), 3) for m, v in reasons["flyable"].items()},
            "vfr_by_month": {MONTHS[m - 1]: round(float(v), 3) for m, v in reasons_vfr["flyable"].items()},
        },
        "main_reason_by_month": {MONTHS[m - 1]: reasons.drop(columns="flyable").loc[m].idxmax() for m in reasons.index},
        "best_2h_slot_by_month": best,
        "p_saturday_or_sunday_has_slot": {MONTHS[m - 1]: round(float(v), 3) for m, v in weekend.items()},
        "forecast_afternoon_from_7am": {
            "target": "2h lesson 14h-16h local within student minimums",
            "train_years": f"{REF_YEARS.start}-{TEST_FROM - 1}", "test_years": f"{TEST_FROM}-{REF_YEARS.stop - 1}",
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in fc.items()
               if k in ("n_train", "n_test", "brier", "brier_climatology", "brier_skill", "auc")},
            "p_afternoon_ok_if_morning_fog_or_mist": round(float(cond.get(True, np.nan)), 3),
            "p_afternoon_ok_otherwise": round(float(cond.get(False, np.nan)), 3),
            "coefficients": {k: round(v, 3) for k, v in fc["coefficients"].items()},
        },
        "stations": map_rows,
    }
    Path("results.json").write_text(json.dumps(results, indent=2, default=str))

    # ---------------------------------------------------------------- figures
    heatmap(table, f"Share of hours within student minimums — {REFERENCE} (ref. for Saint-Cyr), {REF_YEARS.start}–{REF_YEARS.stop - 1}",
            "flyable_month_hour", "% of hours flyable")
    heatmap(slots, "Chance that a 2-hour lesson is flyable from start to finish, by start time",
            "lesson_slots", "% of lessons flyable")

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    labels = {"ceiling": "Ceiling < 2,000 ft", "visibility": "Visibility < 8 km", "crosswind": "Crosswind > 10 kt (rwy 11/29)",
              "wind": "Wind > 15 kt or gusts > 20 kt", "weather": "Fog, snow, thunderstorm…"}
    for (col, lab), c in zip(labels.items(), SERIES):
        ax.plot(range(1, 13), reasons[col] * 100, color=c, linewidth=2, marker="o", markersize=5, label=lab)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, ncol=2)
    style(ax, "Why you stay on the ground: share of daylight hours failing each criterion", "", "% of daylight hours")
    save(fig, "no_fly_reasons")

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    x = np.arange(12)
    ax.bar(x - 0.2, reasons_vfr["flyable"].reindex(range(1, 13)) * 100, width=0.38, color=SERIES[0], label=VFR.name)
    ax.bar(x + 0.2, reasons["flyable"].reindex(range(1, 13)) * 100, width=0.38, color=SERIES[1], label=STUDENT.name)
    ax.bar(x + 0.2, weekend.reindex(range(1, 13)) * 0, width=0)  # keep x range
    ax.plot(x, weekend.reindex(range(1, 13)) * 100, color=INK, marker="D", linewidth=0, markersize=6,
            label="A weekend day has ≥ 1 flyable 2h slot")
    ax.set_xticks(x, MONTHS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    style(ax, "Regulatory VFR vs student minimums, and the odds of flying on a weekend day", "", "%")
    save(fig, "vfr_vs_student")

    # forecast reliability
    p, y = fc["p_test"], fc["y_test"]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    pm = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
    ym = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
    nm = [(idx == b).sum() for b in range(10) if (idx == b).sum() >= 10]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=":")
    ax.scatter(pm, ym, s=[20 + n / 3 for n in nm], color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.plot(pm, ym, color=SERIES[0], linewidth=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, linewidth=0.8)
    style(ax, f"7am METAR → afternoon lesson: forecast vs outcome\n(test {TEST_FROM}–{REF_YEARS.stop - 1}, "
              f"Brier skill {fc['brier_skill']:+.0%} vs climatology)", "Forecast probability", "Observed frequency")
    save(fig, "forecast_reliability")

    # static map for the README
    if map_rows:
        fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
        lo, hi = min(r["annual"] for r in map_rows), max(r["annual"] for r in map_rows)
        sc = ax.scatter([r["lon"] for r in map_rows], [r["lat"] for r in map_rows], c=[r["annual"] * 100 for r in map_rows],
                        cmap="Blues", vmin=lo * 100 - 3, vmax=hi * 100, s=260, edgecolor=INK2, linewidth=0.8, zorder=3)
        offsets = {"LFPN": (-12, -16, "right"), "LFPV": (0, 14, "center"), "LFPO": (0, -20, "center"),
                   "LFPB": (10, -10, "left"), "LFPG": (10, 6, "left"), "LFPM": (12, -4, "left"),
                   "LFPT": (10, 6, "left"), "LFOB": (10, 4, "left"), "LFOE": (0, 14, "center"), "LFOJ": (10, 4, "left")}
        for r in map_rows:
            dx, dy, ha = offsets.get(r["icao"], (10, 4, "left"))
            ax.annotate(f"{r['icao']}  {r['annual'] * 100:.0f}%", (r["lon"], r["lat"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=9, color=INK, ha=ha)
        ax.scatter([HOME["lon"]], [HOME["lat"]], marker="*", s=260, color=SERIES[1], edgecolor=SURFACE, zorder=4)
        ax.annotate("Saint-Cyr (home)", (HOME["lon"], HOME["lat"]), xytext=(-12, 8), textcoords="offset points",
                    fontsize=9, color=SERIES[1], ha="right")
        ax.set_xlim(1.0, 3.1); ax.set_ylim(47.85, 49.6)
        ax.set_aspect(1 / np.cos(np.radians(48.8)))
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, format="%.0f%%")
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=5, integer=True); cb.update_ticks()
        cb.set_label("% of daylight hours flyable", color=INK2); cb.outline.set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.6)
        style(ax, "Aerodromes around Paris: how often a student can fly", "Longitude", "Latitude")
        save(fig, "map_static")
        Path("docs").mkdir(exist_ok=True)
        Path("docs/index.html").write_text(build_map(
            map_rows, HOME, "When can a student pilot fly around Paris?",
            "Share of daylight hours within student minimums (visibility ≥ 8 km, ceiling ≥ 2,000 ft, wind ≤ 15 kt, "
            "crosswind ≤ 10 kt on 11/29, no fog or storm), from METAR archives. Click an aerodrome."), encoding="utf-8")

    if not synthetic:
        write_readme_results(results)
    printable = {k: v for k, v in results.items() if k not in ("stations",)}
    print(json.dumps(printable, indent=1, default=str)[:4000])


def write_readme_results(r: dict, path: str = "README.md"):
    f, fc = r["flyable_share"], r["forecast_afternoon_from_7am"]
    best_m = max(f["student_by_month"], key=f["student_by_month"].get)
    worst_m = min(f["student_by_month"], key=f["student_by_month"].get)
    slots = ", ".join(f"{m} {v['start']} ({v['p']:.0%})" for m, v in r["best_2h_slot_by_month"].items())
    wk = r["p_saturday_or_sunday_has_slot"]
    block = f"""## Results ({r['data']['reference']}, {r['data']['years']}, {r['data']['hours_observed_daylight']:,} daylight hours)

![Flyable hours by month and hour](figures/flyable_month_hour.png)

- **{f['student_annual']:.0%} of daylight hours** meet the student minimums ({f['vfr_annual']:.0%} meet regulatory VFR).
  Best month: **{best_m}** ({f['student_by_month'][best_m]:.0%}), worst: **{worst_m}** ({f['student_by_month'][worst_m]:.0%}).
- **Chance that a given weekend day offers at least one flyable 2-hour slot**: {wk.get('Jun', float('nan')):.0%} in June,
  {wk.get('Dec', float('nan')):.0%} in December.
- **Best 2-hour lesson start by month**: {slots}.
- **Main reason to stay on the ground**, by month: {', '.join(f'{m} {v}' for m, v in r['main_reason_by_month'].items())}.

![Why you stay on the ground](figures/no_fly_reasons.png)
![Lesson slots](figures/lesson_slots.png)
![VFR vs student](figures/vfr_vs_student.png)

### Can the 7am METAR predict a 14h–16h lesson?

Logistic regression on the 7am observation (visibility, ceiling, wind, temperature–dew point spread, fog, precipitation,
season), trained {fc['train_years']} and tested out of sample on {fc['test_years']} ({fc['n_test']} days):
**Brier skill {fc['brier_skill']:+.0%}** against monthly climatology, AUC {fc['auc']:.2f}.
With fog or mist at 7am, the afternoon lesson was flyable {fc['p_afternoon_ok_if_morning_fog_or_mist']:.0%} of the time,
against {fc['p_afternoon_ok_otherwise']:.0%} otherwise.

![Forecast reliability](figures/forecast_reliability.png)

### Map

![Map](figures/map_static.png)

Interactive version: [`docs/index.html`](docs/index.html) (can be served with GitHub Pages)."""
    text = Path(path).read_text(encoding="utf-8")
    s, e = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    i, j = text.index(s) + len(s), text.index(e)
    Path(path).write_text(text[:i] + "\n" + block + "\n" + text[j:], encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
