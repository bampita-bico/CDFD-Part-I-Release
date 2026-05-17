"""
Generate standalone interactive HTML panels for Part I full-stack outputs.

Each panel is self-contained except for local links to the existing PDF/PNG
figures in the same output folder. No internet or JavaScript packages are
required.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

PAPER_ORDER = [
    ("paper_I", "Paper I: Vortex Stability"),
    ("paper_II", "Paper II: Koide and Z3 Vortex Modes"),
    ("paper_III", "Paper III: Topology, Chirality, and Vacuum Density"),
    ("paper_IV", "Paper IV: Vacuum Equation of State"),
    ("paper_V", "Paper V: Zero-Parameter Chirality Phase"),
    ("paper_VI", "Paper VI: Universal Torus Knot Hierarchy"),
    ("paper_VII", "Paper VII: Universal Mass Sum Rule"),
    ("paper_VIII", "Paper VIII: CDFT Torus Knot Spectrum"),
    ("paper_IX", "Paper IX: Even-n Torus Knots"),
    ("paper_X", "Paper X: Public Vacuum Balance"),
    ("paper_XI", "Paper XI: Physics Mysteries and Blancken Layer"),
    ("paper_XII", "Paper XII: Transport Mysteries"),
]

STYLE = r"""
:root{--ink:#1e252b;--muted:#5c6870;--line:#c8d0d6;--bg:#f7f9fb;--blue:#2f6f9f;--green:#4f8f5b;--red:#b9574f}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;color:var(--ink);background:#fff}
header{padding:28px 36px 16px;border-bottom:1px solid var(--line);background:#fbfcfd}h1{margin:0 0 6px;font-size:26px}p{margin:0 0 10px;line-height:1.45}
main{max-width:1240px;margin:0 auto;padding:26px 28px 42px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel,.controls,.note{border:1px solid var(--line);background:var(--bg);border-radius:8px;padding:16px}
.controls{margin-bottom:18px}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}label{display:block;font-size:13px;color:var(--muted);margin:8px 0 4px}
select{width:100%;padding:6px}button{border:1px solid var(--line);background:white;border-radius:6px;padding:7px 10px;cursor:pointer}button.active{background:#22313a;color:#fff}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0}.metrics div{background:white;border:1px solid var(--line);border-radius:6px;padding:10px}.metrics b{display:block;font-size:18px}.metrics span{font-size:12px;color:var(--muted)}
svg{width:100%;height:380px;border:1px solid var(--line);border-radius:8px;background:white}.figure{width:100%;height:520px;border:1px solid var(--line);border-radius:8px;background:white}.figure img{max-width:100%;max-height:100%;display:block;margin:auto}.figure object{width:100%;height:100%}
table{border-collapse:collapse;width:100%;font-size:12px;background:white}th,td{border:1px solid #d7dee2;padding:5px;text-align:left}th{background:#eef3f6}.tablebox{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}.links a{display:inline-block;margin:0 8px 8px 0}
@media(max-width:980px){.grid{grid-template-columns:1fr}.figure{height:420px}.metrics{grid-template-columns:1fr}}
"""

JS = r"""
const W=900,H=380,ML=62,MR=24,MT=24,MB=54;
function fmt(v){ if(v===null||v===''||Number.isNaN(Number(v))) return 'NaN'; const n=Number(v); if(Math.abs(n)>=10000||Math.abs(n)<0.001&&n!==0) return n.toExponential(2); return n.toFixed(4).replace(/\.?0+$/,'');}
function num(v){ const n=Number(v); return Number.isFinite(n)?n:null; }
function clear(n){ while(n.firstChild)n.removeChild(n.firstChild); }
function el(name,attrs={},text=''){ const n=document.createElementNS('http://www.w3.org/2000/svg',name); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if(text)n.textContent=text; return n; }
function metric(items){ const m=document.getElementById('metrics'); m.innerHTML=''; items.forEach(it=>{const d=document.createElement('div'); d.innerHTML=`<b>${it.value}</b><span>${it.label}</span>`; m.appendChild(d);});}
function scales(xs,ys){ const xmin=Math.min(...xs),xmax=Math.max(...xs); let ymin=Math.min(...ys),ymax=Math.max(...ys); if(!Number.isFinite(ymin)){ymin=0;ymax=1} if(ymin===ymax){ymin-=1;ymax+=1} return {x:v=>ML+(v-xmin)/(xmax-xmin||1)*(W-ML-MR),y:v=>H-MB-(v-ymin)/(ymax-ymin)*(H-MT-MB),ymin,ymax};}
function axes(svg,s,xlab,ylab){ svg.appendChild(el('line',{x1:ML,y1:H-MB,x2:W-MR,y2:H-MB,stroke:'#333'})); svg.appendChild(el('line',{x1:ML,y1:MT,x2:ML,y2:H-MB,stroke:'#333'})); for(let i=0;i<5;i++){const y=MT+i*(H-MT-MB)/4,val=s.ymax-(s.ymax-s.ymin)*i/4; svg.appendChild(el('text',{x:ML-9,y:y+4,'text-anchor':'end','font-size':12,fill:'#555'},fmt(val)));} svg.appendChild(el('text',{x:(W+ML-MR)/2,y:H-15,'text-anchor':'middle','font-size':13,fill:'#555'},xlab)); svg.appendChild(el('text',{x:16,y:(H+MT-MB)/2,transform:`rotate(-90 16 ${(H+MT-MB)/2})`,'text-anchor':'middle','font-size':13,fill:'#555'},ylab));}
function plot(rows,xk,yk){ const svg=document.getElementById('chart'); clear(svg); const pts=rows.map((r,i)=>({x:num(r[xk])??i,y:num(r[yk])})).filter(p=>p.y!==null); if(!pts.length){svg.appendChild(el('text',{x:W/2,y:H/2,'text-anchor':'middle',fill:'#666'},'No numeric data for selected columns')); return;} const s=scales(pts.map(p=>p.x),pts.map(p=>p.y)); axes(svg,s,xk,yk); let d=''; pts.forEach((p,i)=>{d+=(i?'L':'M')+s.x(p.x)+' '+s.y(p.y)+' ';}); svg.appendChild(el('path',{d,fill:'none',stroke:'#2f6f9f','stroke-width':3})); pts.forEach(p=>{const c=el('circle',{cx:s.x(p.x),cy:s.y(p.y),r:3,fill:'#b9574f'}); c.appendChild(el('title',{},`${xk}: ${fmt(p.x)}\n${yk}: ${fmt(p.y)}`)); svg.appendChild(c);});}
function table(rows){ const box=document.getElementById('table'); box.innerHTML=''; if(!rows.length){box.textContent='No rows'; return;} const keys=Object.keys(rows[0]); const t=document.createElement('table'); t.innerHTML='<thead><tr>'+keys.map(k=>`<th>${k}</th>`).join('')+'</tr></thead>'; const tb=document.createElement('tbody'); rows.slice(0,500).forEach(r=>{const tr=document.createElement('tr'); tr.innerHTML=keys.map(k=>`<td>${r[k]}</td>`).join(''); tb.appendChild(tr);}); t.appendChild(tb); box.appendChild(t);}
function numericKeys(rows){ if(!rows.length)return []; const keys=Object.keys(rows[0]); return keys.filter(k=>rows.some(r=>num(r[k])!==null)); }
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_panel(folder: str, title: str) -> None:
    path = OUT / folder
    csv_files = sorted(path.glob("*.csv"))
    figure_files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}])
    tables = {p.name: read_csv(p) for p in csv_files}
    figures = [p.name for p in figure_files]
    data = {"tables": tables, "figures": figures}
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head>
<body><header><h1>{title}</h1><p>Interactive local panel for Part I figures, numerical tables, and gate checks.</p><p class="links"><a href="../interactive_index.html">Part I interactive index</a></p></header>
<main><section class="controls"><div class="row"><div style="flex:1"><label>Figure</label><select id="figsel"></select></div><div style="flex:1"><label>Dataset</label><select id="datasel"></select></div><div style="flex:1"><label>X column</label><select id="xsel"></select></div><div style="flex:1"><label>Y column</label><select id="ysel"></select></div></div><div id="metrics" class="metrics"></div></section>
<div class="grid"><section class="panel"><h2>Figure Preview</h2><div id="figbox" class="figure"></div></section><section class="panel"><h2>Numeric Plot</h2><svg id="chart"></svg></section></div>
<section class="panel" style="margin-top:18px"><h2>Data Table</h2><div id="table" class="tablebox"></div></section></main>
<script>const DATA={json.dumps(data)};</script><script>{JS}</script>
<script>
const figsel=document.getElementById('figsel'), datasel=document.getElementById('datasel'), xsel=document.getElementById('xsel'), ysel=document.getElementById('ysel'), figbox=document.getElementById('figbox');
DATA.figures.forEach(f=>{{const o=document.createElement('option'); o.value=f; o.textContent=f; figsel.appendChild(o);}});
Object.keys(DATA.tables).forEach(f=>{{const o=document.createElement('option'); o.value=f; o.textContent=f; datasel.appendChild(o);}});
function renderFigure(){{ const f=figsel.value; figbox.innerHTML=''; if(!f){{figbox.textContent='No figure files found'; return;}} if(f.toLowerCase().endsWith('.pdf')) figbox.innerHTML=`<object data="${{f}}" type="application/pdf"><p><a href="${{f}}">Open ${{f}}</a></p></object>`; else figbox.innerHTML=`<img src="${{f}}" alt="${{f}}">`; }}
function loadColumns(){{ const rows=DATA.tables[datasel.value]||[]; const keys=numericKeys(rows); xsel.innerHTML=''; ysel.innerHTML=''; (keys.length?keys:Object.keys(rows[0]||{{}})).forEach(k=>{{let o=document.createElement('option'); o.value=k; o.textContent=k; xsel.appendChild(o.cloneNode(true)); ysel.appendChild(o);}}); if(keys.length>1) ysel.value=keys[1]; }}
function renderData(){{ const rows=DATA.tables[datasel.value]||[]; metric([{{label:'dataset',value:datasel.value||'none'}},{{label:'rows',value:rows.length}},{{label:'figures',value:DATA.figures.length}}]); table(rows); plot(rows,xsel.value,ysel.value); }}
figsel.onchange=renderFigure; datasel.onchange=()=>{{loadColumns();renderData();}}; xsel.onchange=renderData; ysel.onchange=renderData; renderFigure(); loadColumns(); renderData();
</script></body></html>
"""
    (path / "interactive_panel.html").write_text(html, encoding="utf-8")


def make_index() -> None:
    links = []
    for folder, title in PAPER_ORDER:
        if (OUT / folder).exists():
            links.append(f'<p><a href="{folder}/interactive_panel.html">{title}</a></p>')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Part I Full-Stack Interactive Panels</title><style>{STYLE}</style></head>
<body><header><h1>Part I Full-Stack Interactive Panels</h1><p>Local HTML dashboards for the twelve fundamental-physics papers.</p></header><main><section class="panel">{''.join(links)}</section><section class="note"><p>Panels preview existing figure files and render quick charts from CSV outputs generated by the paper-local scientific Python scripts. They are standalone and offline.</p></section></main></body></html>"""
    (OUT / "interactive_index.html").write_text(html, encoding="utf-8")


def main() -> None:
    for folder, title in PAPER_ORDER:
        if (OUT / folder).exists():
            make_panel(folder, title)
    make_index()
    print(f"Wrote physics interactive panels under {OUT}")


if __name__ == "__main__":
    main()
