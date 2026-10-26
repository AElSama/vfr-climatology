"""Minimal METAR decoder: wind, visibility, ceiling, weather, temperature.

Only the observed part is decoded; the trend (TEMPO, BECMG, NOSIG) and remarks are ignored.
"""
import re

import numpy as np
import pandas as pd

WIND = re.compile(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)$")
VIS_M = re.compile(r"^(\d{4})(NDV)?$")
VIS_SM = re.compile(r"^(P|M)?(\d{1,2})?(?:\s?(\d)/(\d))?SM$")
CLOUD = re.compile(r"^(FEW|SCT|BKN|OVC)(\d{3}|///)(CB|TCU|///)?$")
VERT = re.compile(r"^VV(\d{3}|///)$")
TEMP = re.compile(r"^(M?\d{2})/(M?\d{2})?$")
WX = re.compile(r"^(\+|-|VC)?(MI|PR|BC|DR|BL|SH|TS|FZ)?((?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)*)$")
STOP = {"TEMPO", "BECMG", "NOSIG", "RMK", "TREND"}
MPS_TO_KT = 1.94384


def _temp(s):
    if s is None:
        return np.nan
    return -float(s[1:]) if s.startswith("M") else float(s)


def decode(metar: str) -> dict:
    out = {"wind_dir": np.nan, "wind_kt": np.nan, "gust_kt": np.nan, "vis_m": np.nan,
           "ceiling_ft": np.inf, "cb": False, "wx": "", "temp_c": np.nan, "dew_c": np.nan,
           "cavok": False}
    if not isinstance(metar, str):
        return out
    tokens = metar.replace("=", " ").split()
    wx, seen_vis = [], False
    for i, tok in enumerate(tokens):
        if tok in STOP:
            break
        if i < 3 and (tok in ("METAR", "SPECI", "AUTO", "COR") or re.match(r"^[A-Z]{4}$", tok)
                      or re.match(r"^\d{6}Z$", tok)):
            continue
        if tok in ("AUTO", "COR"):
            continue
        if m := WIND.match(tok):
            k = MPS_TO_KT if m.group(4) == "MPS" else 1.0
            out["wind_dir"] = np.nan if m.group(1) == "VRB" else float(m.group(1))
            out["wind_kt"] = float(m.group(2)) * k
            out["gust_kt"] = float(m.group(3)) * k if m.group(3) else np.nan
            continue
        if tok == "CAVOK":
            out.update(vis_m=10000.0, cavok=True)
            seen_vis = True
            continue
        if not seen_vis and (m := VIS_M.match(tok)):
            v = float(m.group(1))
            out["vis_m"] = 10000.0 if v >= 9999 else v
            seen_vis = True
            continue
        if not seen_vis and tok.endswith("SM") and (m := VIS_SM.match(tok)):
            whole = float(m.group(2) or 0)
            frac = float(m.group(3)) / float(m.group(4)) if m.group(3) else 0.0
            out["vis_m"] = min((whole + frac) * 1609.34, 10000.0)
            seen_vis = True
            continue
        if m := CLOUD.match(tok):
            if m.group(3) in ("CB", "TCU"):
                out["cb"] = True
            if m.group(1) in ("BKN", "OVC") and m.group(2) != "///":
                out["ceiling_ft"] = min(out["ceiling_ft"], float(m.group(2)) * 100)
            continue
        if m := VERT.match(tok):
            out["ceiling_ft"] = min(out["ceiling_ft"], 0.0 if m.group(1) == "///" else float(m.group(1)) * 100)
            continue
        if m := TEMP.match(tok):
            out["temp_c"], out["dew_c"] = _temp(m.group(1)), _temp(m.group(2))
            continue
        if tok.startswith(("R", "Q", "A")) and any(c.isdigit() for c in tok):
            continue  # runway visual range, pressure
        if tok not in ("NSC", "NCD", "SKC", "CLR", "NSW") and (m := WX.match(tok)) and (m.group(2) or m.group(3)):
            wx.append(tok)
    out["wx"] = " ".join(wx)
    return out


def decode_frame(raw: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([decode(m) for m in raw], index=raw.index)
