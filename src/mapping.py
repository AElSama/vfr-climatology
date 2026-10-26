"""Interactive Leaflet map of aerodromes coloured by their share of flyable daylight hours."""
import json

MONTHS = "JFMAMJJASOND"
# sequential blue ramp, light (rarely flyable) to dark (often flyable)
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def colour(p: float, lo: float, hi: float) -> str:
    k = 0 if hi <= lo else int(round((p - lo) / (hi - lo) * (len(RAMP) - 1)))
    return RAMP[max(0, min(len(RAMP) - 1, k))]


def build_map(stations: list[dict], home: dict, title: str, subtitle: str) -> str:
    """stations: dicts with icao, name, lat, lon, annual (0-1), monthly (12 values), n_hours, years."""
    vals = [s["annual"] for s in stations]
    lo, hi = min(vals), max(vals)
    for s in stations:
        s["colour"] = colour(s["annual"], lo, hi)
    payload = json.dumps({"stations": stations, "home": home, "lo": lo, "hi": hi, "ramp": RAMP})
    return TEMPLATE.replace("__DATA__", payload).replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#232322}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:14px 16px 8px}h1{font-size:18px;margin:0 0 2px}header p{margin:0;color:var(--ink2);font-size:13px}
#map{height:calc(100% - 64px);min-height:420px}
.legend{background:var(--card);color:var(--ink);padding:8px 10px;border-radius:8px;border:1px solid var(--line);font-size:12px}
.legend .bar{display:flex;height:10px;margin:6px 0 2px;border-radius:3px;overflow:hidden}
.legend .bar span{flex:1}.legend .ends{display:flex;justify-content:space-between;color:var(--ink2)}
.pop h3{margin:0 0 2px;font-size:14px}.pop .big{font-size:22px;font-weight:600}.pop small{color:#52514e}
</style></head><body>
<header><h1>__TITLE__</h1><p>__SUBTITLE__</p></header><div id="map"></div>
<script>
const D = __DATA__;
const map = L.map('map', {scrollWheelZoom: true}).setView([D.home.lat, D.home.lon], 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 13}).addTo(map);
const M = 'JFMAMJJASOND';
function bars(m){const w=12,h=46,max=1;let s=`<svg width="${12*w}" height="${h+14}" role="img" aria-label="Monthly share of flyable hours">`;
  m.forEach((v,i)=>{const bh=Math.max(1,v/max*h);s+=`<rect x="${i*w+1}" y="${h-bh}" width="${w-3}" height="${bh}" rx="2" fill="#2a78d6"><title>${M[i]}: ${(v*100).toFixed(0)}%</title></rect><text x="${i*w+w/2-1}" y="${h+11}" font-size="9" text-anchor="middle" fill="#52514e">${M[i]}</text>`});
  return s+'</svg>';}
D.stations.forEach(s=>{
  const best = s.monthly.indexOf(Math.max(...s.monthly)), worst = s.monthly.indexOf(Math.min(...s.monthly));
  const html = `<div class="pop"><h3>${s.name} <small>${s.icao}</small></h3>
    <div class="big">${(s.annual*100).toFixed(0)}%</div><small>of daylight hours within student minimums (${s.years})</small><br>
    ${bars(s.monthly)}<br><small>Best ${'January February March April May June July August September October November December'.split(' ')[best]}
    (${(s.monthly[best]*100).toFixed(0)}%), worst ${'January February March April May June July August September October November December'.split(' ')[worst]}
    (${(s.monthly[worst]*100).toFixed(0)}%)</small></div>`;
  L.circleMarker([s.lat, s.lon], {radius: 11, color: '#fcfcfb', weight: 2, fillColor: s.colour, fillOpacity: 0.95})
    .addTo(map).bindPopup(html).bindTooltip(`${s.icao} · ${(s.annual*100).toFixed(0)}%`, {direction: 'top'});
});
L.marker([D.home.lat, D.home.lon]).addTo(map).bindPopup(`<div class="pop"><h3>${D.home.name} <small>${D.home.icao}</small></h3><small>Home field. No METAR: Toussus-le-Noble (LFPN) is used as reference.</small></div>`);
const lg = L.control({position: 'bottomright'});
lg.onAdd = () => {const d = L.DomUtil.create('div', 'legend');
  d.innerHTML = `Share of flyable daylight hours<div class="bar">${D.ramp.map(c=>`<span style="background:${c}"></span>`).join('')}</div>
  <div class="ends"><span>${(D.lo*100).toFixed(0)}%</span><span>${(D.hi*100).toFixed(0)}%</span></div>`; return d;};
lg.addTo(map);
</script></body></html>"""
