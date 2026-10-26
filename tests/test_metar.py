import numpy as np

from src.metar import decode


def test_cavok_day():
    d = decode("LFPN 261030Z 24008KT CAVOK 14/08 Q1018 NOSIG")
    assert d["wind_dir"] == 240 and d["wind_kt"] == 8 and np.isnan(d["gust_kt"])
    assert d["vis_m"] == 10000 and d["cavok"] and d["ceiling_ft"] == np.inf
    assert (d["temp_c"], d["dew_c"]) == (14, 8)


def test_fog_and_vertical_visibility():
    d = decode("LFPN 120630Z 00000KT 0200 R25/0350N FG VV001 03/03 Q1025")
    assert d["vis_m"] == 200 and d["ceiling_ft"] == 100 and d["wx"] == "FG" and d["wind_kt"] == 0


def test_lowest_broken_layer_is_the_ceiling():
    d = decode("LFPO 151200Z 21015G27KT 9999 -RA FEW008 SCT012 BKN025 OVC040 12/10 Q1004")
    assert d["ceiling_ft"] == 2500 and d["gust_kt"] == 27 and d["wx"] == "-RA"


def test_thunderstorm_and_cb():
    d = decode("LFPG 201700Z 25012KT 6000 TSRA SCT030CB BKN050 24/19 Q1012 TEMPO 2000 +TSRA")
    assert d["cb"] and d["wx"] == "TSRA" and d["vis_m"] == 6000 and d["ceiling_ft"] == 5000


def test_trend_is_ignored_and_minus_temperatures():
    d = decode("LFPB 080800Z VRB02KT 3000 BR NSC M02/M03 Q1030 BECMG 9999 NSW")
    assert np.isnan(d["wind_dir"]) and d["vis_m"] == 3000 and d["wx"] == "BR"
    assert d["ceiling_ft"] == np.inf and (d["temp_c"], d["dew_c"]) == (-2, -3)


def test_metres_per_second_and_garbage():
    d = decode("UUEE 010000Z 27005MPS 9999 BKN020 M05/M08 Q1015")
    assert abs(d["wind_kt"] - 9.72) < 0.01
    assert np.isnan(decode(None)["vis_m"])
