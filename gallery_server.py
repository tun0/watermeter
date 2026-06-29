#!/usr/bin/env python3
"""Snapshot gallery server — serves raw/annotated snapshots from SNAPSHOT_DIR.

GET /[?start=YYYYMMDD_HHMMSS&end=YYYYMMDD_HHMMSS&max=N]
  start/end default to (now-24h, now).
  max defaults to GALLERY_MAX_DISPLAY (env) or 50.
  Images at /img/{raw,annotated}/FILENAME.
"""

import bisect
import os
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SNAPSHOT_DIR     = Path(os.environ.get("SNAPSHOT_DIR", "/app/snapshots"))
GALLERY_PORT     = int(os.environ.get("GALLERY_PORT", "8080"))
GALLERY_MAX_DISP = int(os.environ.get("GALLERY_MAX_DISPLAY", "50"))

_SAFE = re.compile(r'^\d{8}_\d{6}\.jpg$')

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>Watermeter snapshots</title>
<style>
  *{box-sizing:border-box}
  body{background:#111;color:#eee;font-family:monospace;margin:0;padding:12px}
  h1{font-size:.95rem;color:#aaa;margin:0 0 10px}
  .ctrl{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  input[type=datetime-local]{background:#222;color:#eee;border:1px solid #444;
    padding:3px 6px;font-family:monospace;color-scheme:dark;font-size:.85rem}
  input[type=number]{background:#222;color:#eee;border:1px solid #444;
    padding:3px 6px;font-family:monospace;width:60px;font-size:.85rem}
  select{background:#222;color:#eee;border:1px solid #444;padding:3px 6px;
    font-family:monospace;font-size:.85rem}
  label{display:flex;align-items:center;gap:5px;cursor:pointer;font-size:.85rem}
  .btn{background:#222;color:#aaa;border:1px solid #444;padding:3px 9px;
    font-family:monospace;cursor:pointer;font-size:.85rem}
  .btn:hover{color:#eee;border-color:#888}
  .btn.primary{border-color:#48f;color:#9cf}
  .btn.primary:hover{background:#1a2a4a;color:#cef}
  #status{color:#666;font-size:.8rem;margin-bottom:8px}
  .grid{display:flex;flex-wrap:wrap;gap:6px}
  .card{background:#1e1e1e;border:2px solid #333;border-radius:4px;
    width:260px;cursor:pointer;position:relative;user-select:none}
  .card:hover{border-color:#555}
  .card.sel{border-color:#f90}
  .card.sel .badge{display:flex}
  .badge{display:none;position:absolute;top:4px;left:4px;background:#f90;color:#111;
    font-weight:bold;font-size:.8rem;border-radius:3px;width:20px;height:20px;
    align-items:center;justify-content:center}
  .card img{width:260px;height:auto;display:block;border-radius:2px 2px 0 0;pointer-events:none}
  .card .ts{padding:3px 6px;font-size:.68rem;color:#777}
  #zoom-bar{display:none;background:#1a1a00;border:1px solid #f90;border-radius:4px;
    padding:6px 10px;margin-bottom:8px;display:none;align-items:center;gap:10px}
  #zoom-bar.show{display:flex}
  #lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
    align-items:center;justify-content:center;z-index:10}
  #lb.open{display:flex}
  #lb img{max-width:95vw;max-height:90vh;border-radius:4px;cursor:zoom-out}
  .nav{position:fixed;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.12);
    border:none;color:#fff;font-size:2rem;padding:.4em .7em;cursor:pointer;
    border-radius:4px;z-index:11;user-select:none}
  #lbp{left:8px} #lbn{right:8px}
  .nav:hover{background:rgba(255,255,255,.25)}
  #lbc{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);
    background:rgba(0,0,0,.65);color:#ccc;font-size:.8rem;
    padding:4px 12px;border-radius:4px;pointer-events:none}
</style>
</head>
<body>
<h1 id="title">Watermeter snapshots</h1>
<div class="ctrl">
  <select id="preset" onchange="applyPreset()">
    <option value="">Custom range</option>
    <option value="1">Last 1 h</option>
    <option value="6">Last 6 h</option>
    <option value="24" selected>Last 24 h</option>
    <option value="168">Last 7 d</option>
  </select>
  <input type="datetime-local" id="start" onchange="clearPreset()">
  <span style="color:#555">–</span>
  <input type="datetime-local" id="end" onchange="clearPreset()">
  <label>Max <input type="number" id="maxn" min="4" max="500" value="/*MAX*/"></label>
  <label><input type="checkbox" id="ann" checked onchange="toggleAnn()"> Annotated</label>
  <button class="btn primary" onclick="go()">Apply</button>
  <button class="btn" onclick="location.reload()">&#8635;</button>
</div>
<div id="zoom-bar">
  <span style="color:#f90">&#9632;</span>
  <span id="zoom-msg"></span>
  <button class="btn primary" id="zoom-btn" onclick="doZoom()">Zoom to selection</button>
  <button class="btn" onclick="clearSel()">Clear</button>
</div>
<div id="status"></div>
<div class="grid" id="grid"></div>

<div id="lb" onclick="closeLb()">
  <button class="nav" id="lbp" onclick="event.stopPropagation();step(-1)">&#8249;</button>
  <img id="lbi" src="" onclick="event.stopPropagation()">
  <button class="nav" id="lbn" onclick="event.stopPropagation();step(+1)">&#8250;</button>
  <div id="lbc"></div>
</div>

<script>
const FILES      = /*FILES*/;
const TOTAL      = /*TOTAL*/;
const INIT_START = "/*INIT_START*/";
const INIT_END   = "/*INIT_END*/";

function parseTs(f) {
  const d=f.slice(0,8), t=f.slice(9,15);
  return new Date(d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6,8)+'T'+
                  t.slice(0,2)+':'+t.slice(2,4)+':'+t.slice(4,6));
}
function toInput(d) {
  return new Date(d-d.getTimezoneOffset()*60000).toISOString().slice(0,16);
}
// datetime-local value → YYYYMMDD_HHMMSS
function toTs(v) { return v.replace(/-/g,'').replace('T','_').replace(':','')+'00'; }

function imgPath(f) {
  return '/img/'+(document.getElementById('ann').checked?'annotated':'raw')+'/'+f;
}

// ── Selection state ────────────────────────────────────────────────────────
const sel = [];   // up to 2 indices into FILES (display order)

function toggleSel(i) {
  const idx = sel.indexOf(i);
  if (idx !== -1) {
    sel.splice(idx, 1);
  } else {
    if (sel.length === 2) {
      // Replace the earlier of the two to allow repositioning
      sel.shift();
    }
    sel.push(i);
    sel.sort((a,b) => a-b);
  }
  renderSel();
}

function renderSel() {
  document.querySelectorAll('.card').forEach((c,i) => {
    const p = sel.indexOf(i);
    c.classList.toggle('sel', p !== -1);
    c.querySelector('.badge').textContent = p !== -1 ? p+1 : '';
  });
  const zbar = document.getElementById('zoom-bar');
  if (sel.length === 2) {
    zbar.classList.add('show');
    // sel is sorted by display index; FILES is newest-first, so sel[1] is older
    document.getElementById('zoom-msg').textContent =
      parseTs(FILES[sel[1]]).toLocaleString() + '  →  ' + parseTs(FILES[sel[0]]).toLocaleString();
  } else {
    zbar.classList.remove('show');
  }
}

function clearSel() { sel.length=0; renderSel(); }

function doZoom() {
  if (sel.length !== 2) return;
  // FILES is newest-first: sel[0] (lower index) = newer = end; sel[1] = older = start
  const p = new URLSearchParams({
    start: FILES[sel[1]].slice(0,15),
    end:   FILES[sel[0]].slice(0,15),
    max:   document.getElementById('maxn').value,
    ann:   document.getElementById('ann').checked ? '1' : '0',
  });
  location.search = '?'+p;
}

// ── Grid rendering ─────────────────────────────────────────────────────────
let lbIdx=0;

function buildGrid() {
  const sampled = TOTAL !== FILES.length;
  document.getElementById('title').textContent =
    'Watermeter snapshots — ' +
    (sampled ? `${FILES.length} of ${TOTAL} shown (equally spaced)` : `${FILES.length} shown`);
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  FILES.forEach((f,i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML =
      '<div class="badge"></div>'+
      '<img src="'+imgPath(f)+'" data-f="'+f+'" loading="lazy">'+
      '<div class="ts">'+parseTs(f).toLocaleString()+'</div>';
    card.addEventListener('click', ()=>toggleSel(i));
    card.querySelector('img').addEventListener('dblclick', e=>{
      e.stopPropagation(); openLb(i);
    });
    grid.appendChild(card);
  });
  document.getElementById('status').textContent =
    FILES.length ? 'Click to select · double-click to enlarge · select 2 to zoom in' : 'No snapshots in this range.';
}

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLb(i) {
  lbIdx=i;
  document.getElementById('lbi').src=imgPath(FILES[i]);
  document.getElementById('lbc').textContent=
    (i+1)+' / '+FILES.length+'  —  '+parseTs(FILES[i]).toLocaleString();
  document.getElementById('lb').classList.add('open');
}
function step(d) { openLb((lbIdx+d+FILES.length)%FILES.length); }
function closeLb() { document.getElementById('lb').classList.remove('open'); }
document.addEventListener('keydown', e=>{
  if (!document.getElementById('lb').classList.contains('open')) return;
  if (e.key==='ArrowRight') step(+1);
  else if (e.key==='ArrowLeft') step(-1);
  else if (e.key==='Escape') closeLb();
});

// ── Annotated toggle ───────────────────────────────────────────────────────
function toggleAnn() {
  const type=document.getElementById('ann').checked?'annotated':'raw';
  document.querySelectorAll('.card img').forEach(img=>{
    img.src='/img/'+type+'/'+img.dataset.f;
  });
  if (document.getElementById('lb').classList.contains('open'))
    document.getElementById('lbi').src=imgPath(FILES[lbIdx]);
}

// ── Navigation controls ────────────────────────────────────────────────────
function applyPreset() {
  const h=parseInt(document.getElementById('preset').value);
  if (!h) return;
  const now=new Date();
  document.getElementById('start').value=toInput(new Date(now-h*3_600_000));
  document.getElementById('end').value=toInput(now);
}
function clearPreset() { document.getElementById('preset').value=''; }
function go() {
  const s=document.getElementById('start').value;
  const e=document.getElementById('end').value;
  if (!s||!e) return;
  const p=new URLSearchParams({start:toTs(s),end:toTs(e),max:document.getElementById('maxn').value,
    ann:document.getElementById('ann').checked?'1':'0'});
  location.search='?'+p;
}

// ── Init ───────────────────────────────────────────────────────────────────
const _qs = new URLSearchParams(location.search);
document.getElementById('start').value=INIT_START;
document.getElementById('end').value=INIT_END;
if (_qs.has('ann')) document.getElementById('ann').checked = _qs.get('ann') !== '0';

buildGrid();
</script>
</body>
</html>
"""


def _now_ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _ts_to_input(ts: str) -> str:
    """YYYYMMDD_HHMMSS → YYYY-MM-DDTHH:MM"""
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}"


def _sample(files: list[str], n: int) -> list[str]:
    """Return n equally-spaced items from files (always includes first and last)."""
    if len(files) <= n:
        return files
    return [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        if path == '/':
            self._serve_gallery(parse_qs(parsed.query))
        elif path.startswith('/img/raw/'):
            self._serve_img(SNAPSHOT_DIR / 'raw', path[9:])
        elif path.startswith('/img/annotated/'):
            self._serve_img(SNAPSHOT_DIR / 'annotated', path[15:])
        else:
            self.send_error(404)

    def _serve_gallery(self, qs: dict):
        now   = _now_ts()
        ago24 = datetime.fromtimestamp(time.time() - 86400).strftime('%Y%m%d_%H%M%S')
        start = qs.get('start', [ago24])[0]
        end   = qs.get('end',   [now])[0]
        max_n = int(qs.get('max', [str(GALLERY_MAX_DISP)])[0])
        max_n = max(4, min(max_n, 500))

        raw_dir = SNAPSHOT_DIR / 'raw'
        # Sorted oldest-first for bisect; filenames are YYYYMMDD_HHMMSS.jpg
        all_files: list[str] = sorted(
            f.name for f in raw_dir.iterdir() if _SAFE.match(f.name)
        ) if raw_dir.is_dir() else []

        # Binary-search slice for [start, end]
        lo = bisect.bisect_left(all_files,  start,    key=lambda f: f[:15])
        hi = bisect.bisect_right(all_files, end,      key=lambda f: f[:15])
        matched = all_files[lo:hi]

        sampled = _sample(matched, max_n)
        # Newest-first for display
        sampled.reverse()

        js_files = '[' + ','.join(f'"{f}"' for f in sampled) + ']'

        html = (
            _HTML
            .replace('/*FILES*/',      js_files)
            .replace('/*TOTAL*/',      str(len(matched)))
            .replace('/*MAX*/',        str(max_n))
            .replace('"/*INIT_START*/"', f'"{_ts_to_input(start)}"')
            .replace('"/*INIT_END*/"',   f'"{_ts_to_input(end)}"')
        ).encode()
        self._respond(200, 'text/html; charset=utf-8', html)

    def _serve_img(self, directory: Path, name: str):
        if not _SAFE.match(name):
            self.send_error(404)
            return
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or not path.is_file():
            self.send_error(404)
            return
        self._respond(200, 'image/jpeg', path.read_bytes())

    def _respond(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    import sys
    if not SNAPSHOT_DIR.is_dir():
        sys.exit(f"SNAPSHOT_DIR not found: {SNAPSHOT_DIR}")
    print(f"Gallery  port={GALLERY_PORT}  dir={SNAPSHOT_DIR}  max={GALLERY_MAX_DISP}", flush=True)
    ThreadingHTTPServer(('', GALLERY_PORT), _Handler).serve_forever()
