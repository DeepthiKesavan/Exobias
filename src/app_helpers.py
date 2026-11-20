from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import visualizations as viz
from .table_utils import arrow_safe

APP_TITLE = "ExoBias – Exoplanet Uncertainty Analysis System"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
FAVICON = PROJECT_ROOT / "assets" / "favicon.png"

# Optional: put your name / guide / institution here and it appears in the sidebar footer.
AUTHOR_LINE = ""

NASA_ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/"
DR25_DOWNLOAD_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_koi&format=csv"
CUMULATIVE_DOWNLOAD_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"

CSS_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#070A16; --surface:#0B1020; --card:#101830; --card2:#172040; --border:#1F2A4A; --border2:#2C3A66;
  --text:#EEF2FF; --muted:#98A5C8; --dim:#66739A; --accent:#6FD6FF; --accent2:#C9D6FF;
  --ok:#4DF0B0; --warn:#FFC857; --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
}
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif; }
.stApp{
  background:
    radial-gradient(1100px 560px at 90% -10%, rgba(111,214,255,.10), transparent 60%),
    radial-gradient(900px 560px at -10% 8%, rgba(139,123,255,.13), transparent 60%),
    radial-gradient(900px 500px at 50% 112%, rgba(255,122,138,.06), transparent 60%),
    var(--bg);
  color:var(--text);
}
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:1180px; padding-top:2.2rem; padding-bottom:4rem; }

/* typography */
.block-container h2{ font-weight:650; letter-spacing:-.01em; margin-top:1.8rem; }
.block-container h3{ font-size:1.12rem; font-weight:600; letter-spacing:.005em; border-left:3px solid var(--accent);
  padding-left:.65rem; margin:1.9rem 0 .8rem; }
.block-container h4{ font-weight:600; color:var(--accent2); }
.block-container p, .block-container li{ line-height:1.62; }
.block-container a{ color:var(--accent); text-decoration:none; }
.block-container a:hover{ text-decoration:underline; }
code{ font-family:var(--mono); font-size:.86em; background:rgba(111,214,255,.09); color:var(--accent2);
  border:1px solid var(--border); border-radius:5px; padding:.08rem .35rem; }
[data-testid="stCaptionContainer"]{ color:var(--muted); }

/* sidebar */
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{ border-radius:8px; color:var(--muted); font-weight:500; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{ background:var(--card); color:var(--text); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{ background:var(--card2); color:var(--text);
  border:1px solid var(--border2); }
.exo-brand{ display:flex; align-items:center; gap:.65rem; margin:.2rem 0 .1rem; }
.exo-brand-name{ background:linear-gradient(90deg,#fff,#9FB6D8); -webkit-background-clip:text; background-clip:text; color:transparent; font-weight:700; letter-spacing:.3em; font-size:1.02rem; color:var(--text); }
.exo-brand-sub{ color:var(--muted); font-size:.74rem; line-height:1.4; margin:.15rem 0 1rem; }
.exo-side-head{ font-size:.68rem; letter-spacing:.16em; color:var(--dim); font-weight:600; margin:1.1rem 0 .45rem; }
.exo-status{ display:flex; align-items:center; gap:.55rem; font-size:.84rem; color:var(--muted); padding:.18rem 0; }
.exo-status.on{ color:var(--text); }
.exo-dot{ width:8px; height:8px; border-radius:50%; background:var(--border2); display:inline-block; flex:0 0 8px; }
.exo-status.on .exo-dot{ background:var(--ok); box-shadow:0 0 0 3px rgba(77,240,176,.16); }
.exo-side-foot{ color:var(--dim); font-size:.72rem; line-height:1.5; margin-top:1.4rem; border-top:1px solid var(--border); padding-top:.8rem; }

/* page header */
.exo-pagehead{ position:relative; display:flex; align-items:center; gap:1.1rem; margin:0 0 1.6rem; padding-bottom:1.2rem; border-bottom:1px solid var(--border);
  animation:rise .6s cubic-bezier(.2,.7,.2,1) both; }
.exo-pagehead::after{ content:""; position:absolute; left:0; bottom:-1px; height:2px; width:140px;
  background:linear-gradient(90deg,var(--accent),transparent); animation:sweep 5s ease-in-out infinite alternate; }
.exo-pagehead-text{ min-width:0; }
.exo-title{ font-size:2.1rem; font-weight:720; letter-spacing:-.02em; margin:0 0 .15rem; line-height:1.15;
  background:linear-gradient(100deg,#FFFFFF 0%,#C9D6FF 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.exo-sub{ color:var(--muted); max-width:780px; font-size:.97rem; margin:0; }
.exo-mini{ position:relative; flex:0 0 54px; width:54px; height:54px; }
.exo-mini i{ position:absolute; left:50%; top:50%; width:16px; height:16px; margin:-8px 0 0 -8px; border-radius:50%;
  background:radial-gradient(circle at 35% 32%, #fff, #FFD9A0 60%, #E9A64B); box-shadow:0 0 14px 3px rgba(255,206,130,.45); }
.exo-mini b{ position:absolute; left:50%; top:50%; border-radius:50%; border:1px solid rgba(111,214,255,.28); animation:spin var(--t) linear infinite; }
.exo-mini b:nth-of-type(1){ width:34px; height:34px; margin:-17px 0 0 -17px; --t:5s; }
.exo-mini b:nth-of-type(2){ width:54px; height:54px; margin:-27px 0 0 -27px; --t:9s; animation-direction:reverse; }
.exo-mini b::after{ content:""; position:absolute; top:-3.5px; left:calc(50% - 3.5px); width:7px; height:7px; border-radius:50%;
  background:var(--p,#6FD6FF); box-shadow:0 0 8px var(--p,#6FD6FF); }
.exo-mini b:nth-of-type(2)::after{ --p:#4DF0B0; }

/* hero */
.exo-hero{ position:relative; overflow:hidden; border:1px solid var(--border2); border-radius:20px; padding:3.2rem 2.8rem 2.8rem;
  background: radial-gradient(ellipse 60% 90% at 84% 40%, rgba(111,214,255,.16), transparent 60%),
              radial-gradient(ellipse 40% 60% at 6% 110%, rgba(139,123,255,.16), transparent 60%),
              linear-gradient(165deg,#0F1630 0%,#0A0F22 62%,#070A16 100%); margin-bottom:1.2rem;
  box-shadow:0 30px 80px -40px rgba(90,120,255,.35); animation:rise .8s cubic-bezier(.2,.7,.2,1) both; }
.exo-hero::before{ content:""; position:absolute; inset:0; pointer-events:none; animation:twinkle 6s ease-in-out infinite;
  background-image:
    radial-gradient(1.9px 1.9px at 8% 18%, rgba(255,255,255,.9) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 17% 64%, rgba(255,255,255,.6) 50%, transparent 52%),
    radial-gradient(2.1px 2.1px at 29% 32%, rgba(190,210,255,.85) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 41% 80%, rgba(255,255,255,.55) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 52% 14%, rgba(255,255,255,.7) 50%, transparent 52%),
    radial-gradient(2px 2px at 63% 58%, rgba(255,255,255,.8) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 74% 26%, rgba(255,255,255,.6) 50%, transparent 52%),
    radial-gradient(1.9px 1.9px at 86% 72%, rgba(190,210,255,.8) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 93% 12%, rgba(255,255,255,.7) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 35% 8%, rgba(255,255,255,.5) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 58% 90%, rgba(255,255,255,.5) 50%, transparent 52%),
    radial-gradient(1.7px 1.7px at 12% 88%, rgba(255,255,255,.45) 50%, transparent 52%); }
.exo-hero-body{ position:relative; z-index:2; }
.exo-hero-title{ font-size:4rem; font-weight:780; letter-spacing:-.03em; line-height:1.02; margin:0 0 .35rem;
  background:linear-gradient(100deg,#FFFFFF 0%,#AEB9CB 35%,#6FD6FF 50%,#AEB9CB 65%,#FFFFFF 100%); background-size:220% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:shimmer 7s linear infinite; }
.exo-hero-sub{ font-size:1.18rem; color:var(--accent2); font-weight:500; margin:0 0 1.1rem; letter-spacing:.01em; }
.exo-hero-text{ color:var(--muted); max-width:600px; font-size:.98rem; line-height:1.68; margin:0 0 1.4rem; }
.exo-hero-meta{ font-size:.78rem; color:var(--dim); letter-spacing:.04em; }

/* orbiting solar system (hero) */
.exo-system{ position:absolute; right:46px; top:50%; width:320px; height:320px; margin-top:-160px; z-index:1; pointer-events:none; }
.exo-sun{ position:absolute; left:50%; top:50%; width:38px; height:38px; margin:-19px 0 0 -19px; border-radius:50%;
  background:radial-gradient(circle at 38% 36%, #FFFFFF 0%, #FFE9B8 38%, #F2B65C 100%);
  box-shadow:0 0 22px 6px rgba(255,214,140,.55), 0 0 70px 24px rgba(255,190,110,.22), 0 0 140px 60px rgba(111,214,255,.10);
  animation:sunpulse 4.5s ease-in-out infinite; }
.exo-orbit{ position:absolute; left:50%; top:50%; width:var(--d); height:var(--d); margin:calc(var(--d) / -2) 0 0 calc(var(--d) / -2);
  border-radius:50%; border:1px solid rgba(111,214,255,.16); animation:spin var(--t) linear infinite; }
.exo-orbit.dash{ border-style:dashed; border-color:rgba(111,214,255,.10); }
.exo-orbit.rev{ animation-direction:reverse; }
.exo-pl{ position:absolute; top:calc(var(--s) / -2); left:calc(50% - var(--s) / 2); width:var(--s); height:var(--s); border-radius:50%;
  background:radial-gradient(circle at 32% 30%, var(--c1) 0%, var(--c2) 58%, var(--c3) 100%);
  box-shadow:inset -4px -4px 9px rgba(0,0,0,.55), 0 0 14px var(--glow); animation:spin var(--t) linear infinite reverse; }
.exo-orbit.rev .exo-pl{ animation-direction:normal; }
.exo-pl.ringed::after{ content:""; position:absolute; left:-62%; right:-62%; top:30%; bottom:30%; border-radius:50%;
  border:2px solid rgba(230,215,180,.55); transform:rotate(-18deg); box-shadow:0 0 8px rgba(230,215,180,.25); }
.exo-moon-orbit{ position:absolute; inset:-15px; border-radius:50%; border:1px dashed rgba(255,255,255,.12); animation:spin 5s linear infinite; }
.exo-moon{ position:absolute; top:-3px; left:calc(50% - 3px); width:6px; height:6px; border-radius:50%; background:#D5DAE3; box-shadow:0 0 6px rgba(255,255,255,.6); }
.exo-comet{ position:absolute; left:0; top:0; width:120px; height:1.5px; background:linear-gradient(90deg,transparent,rgba(190,215,255,.9));
  transform:rotate(28deg); opacity:0; animation:comet 9s ease-in 2s infinite; }
@media (max-width:900px){ .exo-system{ opacity:.28; right:-120px; } .exo-hero-title{ font-size:3rem; } }

/* stat row */
.exo-stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.85rem; margin:1rem 0 .4rem; }
.exo-stat{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1rem 1.15rem; }
.exo-stat-v{ font-size:1.7rem; font-weight:650; letter-spacing:-.01em; color:var(--text); font-family:var(--mono); }
.exo-stat-l{ font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; margin-top:.1rem; }

/* panels and cards */
.exo-panel{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.05rem 1.3rem; margin:.5rem 0 .9rem; }
.exo-panel h4{ margin:0 0 .45rem; font-size:.98rem; color:var(--text); }
.exo-panel p{ margin:0; color:var(--muted); font-size:.93rem; }
.exo-note{ background:rgba(111,214,255,.07); border:1px solid rgba(111,214,255,.25); border-left:3px solid var(--accent);
  border-radius:10px; padding:.85rem 1.1rem; margin:.6rem 0; font-size:.93rem; }
.exo-warn{ background:rgba(255,200,87,.07); border:1px solid rgba(255,200,87,.28); border-left:3px solid var(--warn);
  border-radius:10px; padding:.85rem 1.1rem; margin:.6rem 0; font-size:.93rem; }
.exo-ok{ background:rgba(77,240,176,.07); border:1px solid rgba(77,240,176,.28); border-left:3px solid var(--ok);
  border-radius:10px; padding:.85rem 1.1rem; margin:.6rem 0; font-size:.93rem; }

/* workflow steps */
.exo-steps{ display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin:.4rem 0 .6rem; }
@media (max-width:900px){ .exo-steps{ grid-template-columns:repeat(2,1fr); } }
@media (max-width:600px){ .exo-steps{ grid-template-columns:1fr; } }
.exo-step{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1rem 1.05rem; }
.exo-step-n{ font-family:var(--mono); font-size:.74rem; letter-spacing:.14em; color:var(--accent); }
.exo-step-t{ font-weight:600; margin:.3rem 0 .25rem; font-size:.95rem; }
.exo-step-d{ color:var(--muted); font-size:.83rem; line-height:1.5; }

/* experiment cards */
.exo-exps{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:.9rem; margin:.4rem 0; }
.exo-exp{ background:var(--card); border:1px solid var(--border); border-top:3px solid var(--c,var(--accent)); border-radius:12px; padding:1.1rem 1.2rem; }
.exo-exp-head{ display:flex; align-items:center; gap:.7rem; margin-bottom:.55rem; }
.exo-badge{ width:34px; height:34px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-weight:700;
  background:rgba(255,255,255,.04); border:1px solid var(--c,var(--accent)); color:var(--text); font-family:var(--mono); }
.exo-exp-t{ font-weight:600; font-size:1rem; }
.exo-exp-d{ color:var(--muted); font-size:.89rem; line-height:1.55; }
.exo-exp-f{ margin-top:.7rem; font-family:var(--mono); font-size:.76rem; color:var(--accent2); background:rgba(255,255,255,.03);
  border:1px solid var(--border); border-radius:7px; padding:.4rem .6rem; }

/* architecture strip, chips, buttons */
.exo-arch{ display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin:.5rem 0 .8rem; }
.exo-pill{ background:var(--card2); border:1px solid var(--border2); border-radius:999px; padding:.42rem .95rem; font-size:.84rem; font-family:var(--mono); }
.exo-arrow{ color:var(--dim); }
.exo-chip{ display:inline-block; background:var(--card); border:1px solid var(--border2); color:var(--accent2); border-radius:8px;
  padding:.25rem .7rem; margin:.18rem .25rem .18rem 0; font-size:.82rem; font-weight:500; }
.exo-btn{ display:inline-block; padding:.55rem 1.05rem; border-radius:9px; font-weight:600; font-size:.9rem; margin:.25rem .5rem .25rem 0;
  border:1px solid var(--border2); color:var(--text) !important; background:var(--card2); text-decoration:none !important; }
.exo-btn:hover{ border-color:var(--accent); }
.exo-btn.primary{ background:var(--accent); color:#070A16 !important; border-color:var(--accent); }

/* streamlit widgets */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:.85rem 1rem; }
[data-testid="stMetricLabel"]{ color:var(--muted); }
[data-testid="stMetricValue"]{ color:var(--text); font-family:var(--mono); font-weight:600; }
.stTabs [data-baseweb="tab-list"]{ gap:.2rem; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ color:var(--muted); font-weight:500; padding:.55rem .9rem; }
.stTabs [aria-selected="true"]{ color:var(--text); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--accent); height:2px; }
.stButton > button, .stDownloadButton > button{ border-radius:9px; border:1px solid var(--border2); background:var(--card2);
  color:var(--text); font-weight:500; }
.stButton > button:hover, .stDownloadButton > button:hover{ border-color:var(--accent); color:var(--text); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{ background:var(--accent); color:#070A16; border:1px solid var(--accent); font-weight:600; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{ background:var(--accent2); color:#070A16; }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:10px; overflow:hidden; }
[data-testid="stAlert"]{ border-radius:10px; border:1px solid var(--border); }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:10px; background:var(--card); }
[data-testid="stFileUploader"] section{ background:var(--card); border:1px dashed var(--border2); border-radius:12px; }
hr{ border-color:var(--border); }

/* ===== atmosphere: starfield, drifting nebula, distant planet ===== */
/*STARS*/
.stApp::after{ content:""; position:fixed; right:-230px; bottom:-230px; width:540px; height:540px; border-radius:50%; z-index:0; pointer-events:none;
  background:radial-gradient(circle at 30% 28%, rgba(111,214,255,.13), rgba(50,62,88,.08) 42%, rgba(10,12,16,0) 70%);
  border:1px solid rgba(111,214,255,.10);
  box-shadow:inset -34px -34px 90px rgba(0,0,0,.65), 0 0 0 70px rgba(111,214,255,.018), 0 0 0 71px rgba(111,214,255,.07);
  animation:bob 16s ease-in-out infinite alternate; }
[data-testid="stMain"], section.main{ position:relative; z-index:1; }
[data-testid="stSidebar"]{ z-index:3; }
[data-testid="stMain"]::before{ content:""; position:fixed; top:8vh; left:35vw; width:130px; height:1.5px; z-index:0; pointer-events:none; opacity:0;
  background:linear-gradient(90deg,transparent,rgba(210,225,255,.95)); transform:rotate(24deg); animation:shoot 11s ease-in 3s infinite; }
[data-testid="stMain"]::after{ content:""; position:fixed; top:30vh; left:70vw; width:100px; height:1.5px; z-index:0; pointer-events:none; opacity:0;
  background:linear-gradient(90deg,transparent,rgba(210,225,255,.9)); transform:rotate(24deg); animation:shoot 17s ease-in 9s infinite; }

/* ===== keyframes ===== */
@keyframes spin{ to{ transform:rotate(360deg); } }
@keyframes twinkle{ 0%,100%{ opacity:.35; } 50%{ opacity:1; } }
@keyframes drift{ from{ transform:translate3d(0,0,0); } to{ transform:translate3d(-14px,10px,0); } }
@keyframes bob{ from{ transform:translate3d(0,0,0) rotate(0deg); } to{ transform:translate3d(-26px,-18px,0) rotate(6deg); } }
@keyframes rise{ from{ opacity:0; transform:translateY(16px); } to{ opacity:1; transform:none; } }
@keyframes shimmer{ from{ background-position:0% 0; } to{ background-position:-220% 0; } }
@keyframes sunpulse{ 0%,100%{ transform:scale(1); } 50%{ transform:scale(1.07); } }
@keyframes sweep{ from{ width:80px; } to{ width:320px; } }
@keyframes shoot{ 0%,92%{ opacity:0; margin:0 0 0 0; } 94%{ opacity:1; } 100%{ opacity:0; margin:260px 0 0 -420px; } }
@keyframes comet{ 0%,70%{ opacity:0; transform:translate(0,0) rotate(28deg); } 75%{ opacity:1; } 100%{ opacity:0; transform:translate(330px,175px) rotate(28deg); } }
@keyframes flow{ 0%,22%,100%{ border-color:var(--border2); box-shadow:none; color:var(--text); }
  9%{ border-color:var(--accent); box-shadow:0 0 18px rgba(111,214,255,.45); color:#fff; } }
@keyframes arrowflow{ 0%,22%,100%{ color:var(--dim); } 10%{ color:var(--accent); } }
@keyframes ping{ 0%{ box-shadow:0 0 0 0 rgba(77,240,176,.5); } 100%{ box-shadow:0 0 0 9px rgba(77,240,176,0); } }
@keyframes glowline{ from{ transform:translateX(-100%); } to{ transform:translateX(100%); } }

/* ===== motion on components ===== */
.block-container{ animation:fadein .55s ease both; }
@keyframes fadein{ from{ opacity:0; } to{ opacity:1; } }
.exo-stat, .exo-step, .exo-exp, .exo-panel, .exo-note, .exo-warn, .exo-ok{ animation:rise .7s cubic-bezier(.2,.7,.2,1) backwards; animation-delay:calc(var(--i,0) * 80ms);
  transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease, background .25s ease; }
.exo-stat:hover, .exo-step:hover, .exo-exp:hover, .exo-panel:hover{ transform:translateY(-4px); border-color:rgba(111,214,255,.45);
  box-shadow:0 14px 34px -16px rgba(90,140,255,.45); }
.exo-exp:hover{ border-color:var(--c); box-shadow:0 14px 34px -14px var(--c); }
.exo-stat{ position:relative; overflow:hidden; }
.exo-stat::after{ content:""; position:absolute; left:0; top:0; height:2px; width:100%;
  background:linear-gradient(90deg,transparent,var(--accent),transparent); animation:glowline 4.5s ease-in-out infinite; animation-delay:calc(var(--i,0) * .6s); }
.exo-step-n{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:50%;
  border:1px solid rgba(111,214,255,.5); background:rgba(111,214,255,.08); }
.exo-step:hover .exo-step-n{ background:var(--accent); color:#070A16; }
.exo-badge{ box-shadow:0 0 0 0 var(--c); transition:box-shadow .3s ease, background .3s ease; }
.exo-exp:hover .exo-badge{ background:var(--c); box-shadow:0 0 18px -2px var(--c); }
.exo-pill{ animation:flow 7s ease-in-out infinite; animation-delay:calc(var(--i,0) * .9s); }
.exo-arrow{ animation:arrowflow 7s ease-in-out infinite; animation-delay:calc(var(--i,0) * .9s + .45s); }
.exo-chip{ transition:transform .2s ease, border-color .2s ease, color .2s ease; }
.exo-chip:hover{ transform:translateY(-3px); border-color:var(--accent); color:#fff; }
.exo-status.on .exo-dot{ animation:ping 2.2s ease-out infinite; }
.exo-logo{ position:relative; width:32px; height:32px; flex:0 0 32px; }
.exo-logo i{ position:absolute; left:50%; top:50%; width:12px; height:12px; margin:-6px 0 0 -6px; border-radius:50%;
  background:radial-gradient(circle at 35% 32%, #fff, #C9D6FF 60%, #7F93B5); box-shadow:0 0 10px rgba(111,214,255,.6); }
.exo-logo b{ position:absolute; inset:0; border-radius:50%; border:1px solid rgba(111,214,255,.45); animation:spin 6s linear infinite; }
.exo-logo b::after{ content:""; position:absolute; top:-3px; left:calc(50% - 3px); width:6px; height:6px; border-radius:50%; background:#EEF2FF; box-shadow:0 0 7px #fff; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{ transition:transform .2s ease, background .2s ease, color .2s ease; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{ transform:translateX(4px); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{ box-shadow:inset 3px 0 0 var(--accent), 0 0 20px -8px var(--accent); }
.stButton > button, .stDownloadButton > button{ transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover{ transform:translateY(-2px); box-shadow:0 8px 22px -10px rgba(111,214,255,.6); }
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"]{ box-shadow:0 0 0 0 rgba(111,214,255,.55); }
button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover{ box-shadow:0 8px 28px -6px rgba(111,214,255,.75); }
[data-testid="stMetric"]{ transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation:rise .6s ease both; }
[data-testid="stMetric"]:hover{ transform:translateY(-3px); border-color:rgba(111,214,255,.45); box-shadow:0 12px 28px -16px rgba(90,140,255,.5); }
[data-testid="stDataFrame"], [data-testid="stImage"] img, [data-testid="stPyplot"] img{ transition:box-shadow .3s ease, border-color .3s ease; }
[data-testid="stDataFrame"]:hover{ border-color:rgba(111,214,255,.4); box-shadow:0 10px 30px -18px rgba(90,140,255,.5); }
.block-container h3{ position:relative; }
.block-container h3::before{ content:""; position:absolute; left:-3px; top:0; bottom:0; width:3px; background:var(--accent); box-shadow:0 0 12px var(--accent); }
.stTabs [data-baseweb="tab"]{ transition:color .2s ease, background .2s ease; }
.stTabs [data-baseweb="tab"]:hover{ color:var(--text); background:rgba(111,214,255,.06); }

/* ===== spectral palette: starlight, atmospheres, light spectra ===== */
:root{ --violet:#8B7BFF; --cyan:#6FD6FF; --mint:#4DF0B0; --amber:#FFC857; --coral:#FF7A8A;
  --spectrum:linear-gradient(90deg,#8B7BFF 0%,#6FD6FF 28%,#4DF0B0 52%,#FFC857 76%,#FF7A8A 100%); }
.exo-pagehead::after{ background:var(--spectrum); box-shadow:0 0 14px rgba(111,214,255,.55); }
.exo-hero{ border-color:rgba(111,214,255,.22); box-shadow:0 30px 90px -40px rgba(139,123,255,.55), 0 0 0 1px rgba(139,123,255,.06) inset; }
.exo-hero::after{ content:""; position:absolute; left:0; right:0; bottom:0; height:2px; background:var(--spectrum); opacity:.85;
  box-shadow:0 0 18px rgba(111,214,255,.6); }
.exo-hero-title{ background:linear-gradient(100deg,#FFFFFF 0%,#C9D6FF 30%,#6FD6FF 48%,#8B7BFF 58%,#C9D6FF 72%,#FFFFFF 100%); background-size:220% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.exo-title{ background:linear-gradient(100deg,#FFFFFF 0%,#C9D6FF 55%,#9FE6FF 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.exo-stat::after{ background:var(--spectrum); }
.exo-stat-v{ color:#fff; text-shadow:0 0 18px rgba(111,214,255,.45); }
.exo-eyebrow, .exo-step-n{ color:var(--cyan); }
.block-container h3{ border-left:0; padding-left:.9rem; }
.block-container h3::before{ background:linear-gradient(180deg,var(--violet),var(--cyan),var(--mint)); box-shadow:0 0 14px rgba(111,214,255,.7); width:3px; border-radius:2px; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{ background:linear-gradient(100deg,#6FD6FF,#8B7BFF); border:0; color:#050816; font-weight:650; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{ background:linear-gradient(100deg,#9BE7FF,#A89BFF); color:#050816; }
.exo-btn.primary{ background:linear-gradient(100deg,#6FD6FF,#8B7BFF); border:0; color:#050816 !important; }
.exo-note{ border-left-color:var(--cyan); }
.exo-pill{ background:linear-gradient(180deg,#172040,#101830); }
.exo-chip:hover{ box-shadow:0 0 16px -4px var(--cyan); }
.exo-status.on .exo-dot{ background:var(--mint); box-shadow:0 0 10px var(--mint); }
.exo-brand-name{ background:var(--spectrum); -webkit-background-clip:text; background-clip:text; color:transparent; }
.exo-sun{ box-shadow:0 0 22px 6px rgba(255,214,140,.6), 0 0 80px 28px rgba(255,150,120,.22), 0 0 160px 70px rgba(139,123,255,.16); }
.exo-orbit{ border-color:rgba(111,214,255,.2); }
.stApp::after{ background:radial-gradient(circle at 30% 28%, rgba(139,123,255,.22), rgba(60,80,160,.12) 42%, rgba(7,10,22,0) 70%); border-color:rgba(139,123,255,.16); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--spectrum); }

/* ===== living background: drifting nebula + rotating galaxy swirl ===== */
.stApp{ background-size:135% 135%; background-repeat:no-repeat; animation:nebula 46s ease-in-out infinite alternate; }
@keyframes nebula{ from{ background-position:0% 0%; } to{ background-position:100% 100%; } }
[data-testid="stAppViewContainer"]::after{ content:""; position:fixed; left:-420px; top:-420px; width:980px; height:980px; border-radius:50%;
  z-index:0; pointer-events:none; opacity:.16; filter:blur(22px); animation:spin 150s linear infinite;
  background:conic-gradient(from 0deg, transparent 0%, rgba(139,123,255,.95) 12%, transparent 26%, rgba(111,214,255,.85) 42%,
    transparent 56%, rgba(255,122,138,.65) 74%, transparent 88%);
  -webkit-mask-image:radial-gradient(circle, transparent 8%, #000 36%, transparent 72%);
  mask-image:radial-gradient(circle, transparent 8%, #000 36%, transparent 72%); }

/* ===== image gallery ===== */
[data-testid="stImage"]{ overflow:hidden; border-radius:16px; border:1px solid var(--border2); background:#050816;
  box-shadow:0 18px 50px -28px rgba(111,214,255,.45); transition:border-color .3s ease, box-shadow .3s ease, transform .3s ease; }
[data-testid="stImage"]:hover{ border-color:rgba(111,214,255,.55); box-shadow:0 20px 60px -24px rgba(139,123,255,.7); transform:translateY(-3px); }
[data-testid="stImage"] img{ transition:transform .9s cubic-bezier(.2,.7,.2,1); border-radius:0; }
[data-testid="stImage"]:hover img{ transform:scale(1.07); }
[data-testid="stImageCaption"], [data-testid="stImage"] figcaption{ color:var(--muted); font-size:.8rem; }

/* ===== results leaderboard ===== */
.exo-lb{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.3rem; margin:.5rem 0 .9rem; }
.exo-lb-title{ font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:.8rem; }
.exo-lb-row{ display:grid; grid-template-columns:minmax(150px,230px) 1fr 70px; align-items:center; gap:.9rem; margin:.5rem 0; }
.exo-lb-name{ font-size:.86rem; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.exo-lb-name small{ color:var(--muted); }
.exo-lb-track{ height:12px; background:rgba(255,255,255,.05); border-radius:999px; overflow:hidden; }
.exo-lb-bar{ height:100%; width:var(--w); border-radius:999px; background:var(--c);
  box-shadow:0 0 14px -2px var(--c); transform-origin:left; animation:growbar 1.4s cubic-bezier(.2,.8,.2,1) backwards; animation-delay:calc(var(--i,0) * 110ms); }
.exo-lb-val{ font-family:var(--mono); font-size:.86rem; text-align:right; color:#fff; }
@keyframes growbar{ from{ transform:scaleX(0); } to{ transform:scaleX(1); } }
.exo-empty{ border:1px dashed var(--border2); border-radius:14px; padding:1.6rem 1.4rem; text-align:center; color:var(--muted);
  background:linear-gradient(180deg,rgba(139,123,255,.06),rgba(111,214,255,.03)); }
.exo-empty b{ color:var(--text); }
@media (prefers-reduced-motion:reduce){
  .stApp *, .stApp::before, .stApp::after, [data-testid="stAppViewContainer"]::before, [data-testid="stMain"]::before, [data-testid="stAppViewContainer"]::after,
  [data-testid="stMain"]::after{ animation:none !important; transition:none !important; }
}
</style>
"""



def _star_css() -> str:
    """Two fixed star layers (twinkling / drifting). Positions come from a seeded RNG so the sky is identical on every page."""
    import random
    rng = random.Random(11)

    def layer(n: int, colours: list[str]) -> str:
        return ",".join(f"{rng.uniform(0, 100):.1f}vw {rng.uniform(0, 100):.1f}vh 0 {rng.choice(colours)}" for _ in range(n))

    small = layer(120, ["rgba(255,255,255,.75)", "rgba(190,210,255,.7)", "rgba(255,255,255,.45)"])
    large = layer(38, ["rgba(255,255,255,.95)", "rgba(170,200,255,.9)", "rgba(255,235,200,.85)"])
    return (
        ".stApp::before{content:\"\";position:fixed;top:0;left:0;width:1px;height:1px;z-index:0;pointer-events:none;"
        f"box-shadow:{small};animation:twinkle 5s ease-in-out infinite;}}\n"
        "[data-testid=\"stAppViewContainer\"]::before{content:\"\";position:fixed;top:0;left:0;width:2px;height:2px;border-radius:50%;"
        f"z-index:0;pointer-events:none;box-shadow:{large};animation:twinkle 8s ease-in-out infinite reverse, drift 40s ease-in-out infinite alternate;}}"
    )


CSS = CSS_BASE.replace("/*STARS*/", _star_css())

SCORE_WARNING = (
    "<b>koi_score is a Robovetter vetting score, not a calibrated probability that an object is a real planet.</b> "
    "Experiments B and C use it only as a filter / training weight: they are sensitivity analyses to be tested, not assumed to help."
)
SELECTION_WARNING = (
    "The KOI table is an <b>observed, detection-selected sample</b>. Differences between its distributions are not "
    "differences in the intrinsic exoplanet population, and nothing here estimates occurrence rates."
)

_LOGO_HTML = '<div class="exo-logo"><i></i><b></b></div>'


# --------------------------------------------------------------------------------------
# Page scaffolding
# --------------------------------------------------------------------------------------
def setup_page(title: str, icon: str | None = None) -> None:
    """Page config, global stylesheet and the branded sidebar. ``icon`` is accepted for backwards compatibility."""
    page_icon = None
    try:
        from PIL import Image
        if FAVICON.exists():
            page_icon = Image.open(FAVICON)
    except Exception:  # favicon is cosmetic; never block the page
        page_icon = None
    st.set_page_config(page_title=f"{title} | ExoBias", page_icon=page_icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            f'<div class="exo-brand">{_LOGO_HTML}<span class="exo-brand-name">EXOBIAS</span></div>'
            '<div class="exo-brand-sub">Uncertainty-aware machine learning for Kepler KOI candidate labels</div>',
            unsafe_allow_html=True,
        )
        raw = st.session_state.get("raw_df")
        items = [
            (raw is not None, "Dataset loaded" + (f" · {st.session_state.get('source_name')}" if raw is not None else "")),
            (st.session_state.get("data") is not None, "Data preprocessed"),
            (bool(st.session_state.get("results")), "Experiments trained"),
            (bool(st.session_state.get("population")), "Population analysis"),
            (bool(st.session_state.get("uncertainty")), "Robustness analysis"),
        ]
        rows = "".join(
            f'<div class="exo-status{" on" if on else ""}"><span class="exo-dot"></span>{label}</div>' for on, label in items
        )
        st.markdown(f'<div class="exo-side-head">SESSION PROGRESS</div>{rows}', unsafe_allow_html=True)
        if AUTHOR_LINE:
            st.markdown(f'<div class="exo-side-foot">{AUTHOR_LINE}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", step: int | None = None, area: str = "") -> None:
    """Page title block. ``step`` and ``area`` are accepted for backwards compatibility but no longer displayed."""
    st.markdown(
        '<div class="exo-pagehead"><div class="exo-mini"><i></i><b></b><b></b></div><div class="exo-pagehead-text">'
        f'<div class="exo-title">{title}</div>'
        + (f'<p class="exo-sub">{subtitle}</p>' if subtitle else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="exo-{kind}">{text}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="exo-panel"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Overview components
# --------------------------------------------------------------------------------------
_SYSTEM_HTML = (
    '<div class="exo-system">'
    '<div class="exo-sun"></div>'
    '<div class="exo-orbit dash" style="--d:96px;--t:14s"><span class="exo-pl" style="--s:11px;--t:14s;--c1:#EEF2FF;--c2:#8E98A8;--c3:#4B5362;--glow:rgba(200,210,225,.35)"></span></div>'
    '<div class="exo-orbit" style="--d:156px;--t:24s"><span class="exo-pl" style="--s:16px;--t:24s;--c1:#FFE3B0;--c2:#FFC857;--c3:#7A5416;--glow:rgba(255,200,87,.45)">'
    '<span class="exo-moon-orbit"><span class="exo-moon"></span></span></span></div>'
    '<div class="exo-orbit dash rev" style="--d:216px;--t:38s"><span class="exo-pl ringed" style="--s:22px;--t:38s;--c1:#C9F2E3;--c2:#4DF0B0;--c3:#1B5A47;--glow:rgba(77,240,176,.45)"></span></div>'
    '<div class="exo-orbit" style="--d:268px;--t:56s"><span class="exo-pl" style="--s:18px;--t:56s;--c1:#D8E4FF;--c2:#5C86D6;--c3:#1E3566;--glow:rgba(92,134,214,.5)"></span></div>'
    '<div class="exo-orbit dash" style="--d:314px;--t:80s"><span class="exo-pl" style="--s:12px;--t:80s;--c1:#E3D6FF;--c2:#8E72D8;--c3:#3A2A73;--glow:rgba(142,114,216,.45)"></span></div>'
    '<div class="exo-comet"></div>'
    "</div>"
)


def hero(title: str, subtitle: str, text: str, eyebrow: str = "", meta: str = "") -> None:
    st.markdown(
        '<div class="exo-hero">' + _SYSTEM_HTML + '<div class="exo-hero-body">'
        + (f'<div class="exo-hero-eyebrow">{eyebrow}</div>' if eyebrow else "")
        + f'<div class="exo-hero-title">{title}</div>'
        f'<div class="exo-hero-sub">{subtitle}</div>'
        f'<p class="exo-hero-text">{text}</p>'
        + (f'<div class="exo-hero-meta">{meta}</div>' if meta else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )



def leaderboard(table: pd.DataFrame, metric: str = "f1_macro", title: str = "", top: int = 9) -> None:
    """Animated horizontal bars for one metric (computed results only). Bar colour = training strategy."""
    from .visualizations import STRATEGY_COLORS
    t = table.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top)
    if t.empty:
        return
    hi = max(float(t[metric].max()), 1e-9)
    rows = "".join(
        f'<div class="exo-lb-row" style="--i:{i}"><div class="exo-lb-name">{r.model} <small>· {r.strategy_key}</small></div>'
        f'<div class="exo-lb-track"><div class="exo-lb-bar" style="--w:{max(float(r[metric]) / hi * 100, 2):.1f}%;--c:{STRATEGY_COLORS.get(r.strategy_key, "#6FD6FF")};--i:{i}"></div></div>'
        f'<div class="exo-lb-val">{float(r[metric]):.3f}</div></div>'
        for i, (_, r) in enumerate(t.iterrows())
    )
    st.markdown(f'<div class="exo-lb"><div class="exo-lb-title">{title or metric}</div>{rows}</div>', unsafe_allow_html=True)


def results_showcase() -> None:
    """Overview panel: shows the session's real results if experiments have been trained, otherwise an honest placeholder."""
    from .experiments import results_to_frame
    results = st.session_state.get("results")
    if not results:
        st.markdown('<div class="exo-empty"><b>No results yet.</b><br>Train experiments in the ANN Workbench and this panel fills '
                    "with your measured results. Nothing is shown until it has actually been computed.</div>", unsafe_allow_html=True)
        return
    full = results_to_frame(results, "full")
    c1, c2 = st.columns(2)
    with c1:
        leaderboard(full, "f1_macro", "Macro-F1 on the common test set (higher is better)")
    with c2:
        leaderboard(full, "accuracy", "Accuracy on the common test set (higher is better)")
    st.caption("Bar colours show the training strategy: blue = A baseline, orange = B filtered, green = C weighted. Single seed and split; see Uncertainty Analysis for stability.")


def gallery(items: list[tuple[str, str]], columns: int | None = None) -> None:
    """Show local images (path under assets/images, caption) in a row."""
    cols = st.columns(columns or len(items))
    for col, (name, caption) in zip(cols, items):
        path = PROJECT_ROOT / "assets" / "images" / name
        if path.exists():
            col.image(str(path), caption=caption, use_container_width=True)


def stat_row(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="exo-stat" style="--i:{i}"><div class="exo-stat-v">{v}</div><div class="exo-stat-l">{l}</div></div>' for i, (v, l) in enumerate(items))
    st.markdown(f'<div class="exo-stats">{html}</div>', unsafe_allow_html=True)


def steps_flow(steps: list[tuple[str, str]]) -> None:
    html = "".join(
        f'<div class="exo-step" style="--i:{i}"><div class="exo-step-n">{i:02d}</div><div class="exo-step-t">{t}</div><div class="exo-step-d">{d}</div></div>'
        for i, (t, d) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="exo-steps">{html}</div>', unsafe_allow_html=True)


def experiment_cards(cards: list[dict]) -> None:
    """Each dict: letter, title, body, formula, color."""
    html = "".join(
        f'<div class="exo-exp" style="--c:{c["color"]};--i:{i}"><div class="exo-exp-head"><div class="exo-badge">{c["letter"]}</div>'
        f'<div class="exo-exp-t">{c["title"]}</div></div><div class="exo-exp-d">{c["body"]}</div>'
        f'<div class="exo-exp-f">{c["formula"]}</div></div>'
        for i, c in enumerate(cards)
    )
    st.markdown(f'<div class="exo-exps">{html}</div>', unsafe_allow_html=True)


def architecture_strip(parts: list[str]) -> None:
    pieces = []
    for i, label in enumerate(parts):
        if i:
            pieces.append(f'<span class="exo-arrow" style="--i:{i - 1}">→</span>')
        pieces.append(f'<span class="exo-pill" style="--i:{i}">{label}</span>')
    st.markdown(f'<div class="exo-arch">{"".join(pieces)}</div>', unsafe_allow_html=True)


def chips(labels: list[str]) -> None:
    st.markdown("".join(f'<span class="exo-chip">{l}</span>' for l in labels), unsafe_allow_html=True)


def dataset_guide(compact: bool = False) -> None:
    """How to obtain the dataset to upload, with direct download links."""
    st.markdown(
        '<div class="exo-panel"><h4>Dataset to upload: Kepler DR25 KOI table</h4>'
        "<p>Source: NASA Exoplanet Archive, table <code>q1_q17_dr25_koi</code> (Kepler Objects of Interest, "
        "Q1–Q17 DR25 delivery). Download it as CSV and upload it below.</p>"
        f'<div style="margin-top:.7rem"><a class="exo-btn primary" href="{DR25_DOWNLOAD_URL}" target="_blank">Download DR25 KOI table (CSV)</a>'
        f'<a class="exo-btn" href="{NASA_ARCHIVE_URL}" target="_blank">Open NASA Exoplanet Archive</a></div></div>',
        unsafe_allow_html=True,
    )
    if compact:
        return
    st.markdown(
        "1. Click **Download DR25 KOI table (CSV)** (the file can be large; wait for the download to finish).\n"
        "2. If the saved file has no `.csv` extension, rename it, for example to `koi_dr25.csv`.\n"
        "3. Upload it using the file box below, or place it in the project's `data/` folder.\n\n"
        "If the direct link does not work in your browser, open the archive, choose **Data → Kepler Objects of Interest**, "
        "select the **Q1–Q17 DR25** table, and use **Download Table → CSV (all columns)**."
    )
    st.caption(
        f"Optional, for the three-class experiment only: the archive's cumulative KOI table also contains CONFIRMED objects "
        f"([cumulative table CSV]({CUMULATIVE_DOWNLOAD_URL})). It is a different catalogue version than DR25, so treat it as a separate source."
    )


# --------------------------------------------------------------------------------------
# State guards
# --------------------------------------------------------------------------------------
def init_state() -> None:
    for key, default in (("figures", {}), ("results", {}), ("population", {}), ("uncertainty", {})):
        if key not in st.session_state:
            st.session_state[key] = default


def require_raw() -> pd.DataFrame:
    init_state()
    df = st.session_state.get("raw_df")
    if df is None:
        st.warning("No dataset loaded yet. Go to **Data Explorer** and upload a NASA KOI CSV file.")
        st.stop()
    return df


def require_data():
    require_raw()
    data = st.session_state.get("data")
    if data is None:
        st.warning("Data has not been preprocessed yet. Go to **Data Preprocessing** and click *Run preprocessing*.")
        st.stop()
    return data


def require_results() -> dict:
    require_data()
    results = st.session_state.get("results")
    if not results:
        st.warning("No experiments have been trained yet. Go to **ANN Workbench** and click *Train selected experiments*.")
        st.stop()
    return results


def show_validation(report) -> None:
    for kind, message in report.as_messages():
        {"error": st.error, "warning": st.warning, "info": st.info}[kind](message)


# --------------------------------------------------------------------------------------
# Figures and tables
# --------------------------------------------------------------------------------------
def show_fig(fig, name: str, key: str | None = None, download: bool = True) -> None:
    """Render a Matplotlib figure, register it for the Results page and offer a PNG download."""
    png = viz.fig_to_png(fig)
    st.session_state.setdefault("figures", {})[name] = png
    st.pyplot(fig)
    plt.close(fig)
    if download:
        st.download_button("Download PNG", png, file_name=f"{name.replace(' ', '_').lower()}.png", mime="image/png",
                           key=f"dl_{key or name}")


def show_df(df: pd.DataFrame, **kwargs) -> None:
    """st.dataframe that never fails on mixed-type columns (display-only conversion)."""
    st.dataframe(arrow_safe(df), **kwargs)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_table(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, csv_bytes(df), file_name=filename, mime="text/csv", key=key or f"dlcsv_{filename}")


def fmt_table(df: pd.DataFrame, digits: int = 4):
    """Dataframe with floats rounded for display (the underlying data is untouched)."""
    return df.round(digits)


def show_cohort_info(info: dict) -> None:
    """Training-cohort summary for one experiment (always shows the training-size difference)."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows used", f"{info['n_train']:,}", delta=f"{-info['n_train_removed']:,} vs baseline" if info["n_train_removed"] else None)
    c2.metric("Rows excluded from training", f"{info['n_train_removed']:,}")
    c3.metric("Validation rows used", f"{info['n_val']:,} / {info['n_val_before']:,}")
    st.write("Rows removed per class:", info["removed_by_class"], "· training class counts:", info["train_class_counts"])
    if info.get("weights_summary"):
        w = info["weights_summary"]
        st.write(f"Sample weights (train): mean {w['mean']:.3f}, min {w['min']:.3f}, max {w['max']:.3f}, "
                 f"effective sample size {w['effective_sample_size']:.0f}")
    for n in info.get("notes", []):
        note(n, "warn" if "SMALLER" in n or "NOT calibrated" in n else "note")


def show_model_detail(m, data, key: str) -> None:
    """Metrics, confusion matrix, classification report and calibration for one trained model."""
    from .evaluation import (classification_report_df, confusion_matrix_df, expected_calibration_error,
                             reliability_curve_data)
    met = m.metrics
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{met['accuracy']:.4f}")
    cols[1].metric("Macro precision", f"{met['precision_macro']:.4f}")
    cols[2].metric("Macro recall", f"{met['recall_macro']:.4f}")
    cols[3].metric("Macro-F1", f"{met['f1_macro']:.4f}")
    cols[4].metric("Log loss", f"{met['log_loss']:.4f}")
    cols = st.columns(5)
    cols[0].metric("Brier", f"{met['brier']:.4f}")
    cols[1].metric("ECE (10 bins)", f"{met['ece']:.4f}")
    cols[2].metric("ROC-AUC", "n/a" if met["roc_auc"] != met["roc_auc"] else f"{met['roc_auc']:.4f}")
    cols[3].metric("PR-AUC", "n/a" if met["pr_auc"] != met["pr_auc"] else f"{met['pr_auc']:.4f}")
    cols[4].metric("Test rows", f"{met['n_samples']:,}")
    if m.metrics_hc:
        st.caption(f"On the high-confidence part of the test set ({m.metrics_hc['n_samples']:,} rows): "
                   f"accuracy {m.metrics_hc['accuracy']:.4f}, macro-F1 {m.metrics_hc['f1_macro']:.4f}, "
                   f"log loss {m.metrics_hc['log_loss']:.4f}, ECE {m.metrics_hc['ece']:.4f}.")
    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix_df(data.y_test, m.proba_test, data.class_names)
        show_fig(viz.fig_confusion_matrix(cm, f"{m.label}"), f"confusion_{m.strategy}_{m.model_name}", f"{key}_cm")
    with c2:
        st.markdown("**Classification report**")
        show_df(fmt_table(classification_report_df(data.y_test, m.proba_test, data.class_names), 3))
        ece, bins = expected_calibration_error(data.y_test, m.proba_test)
        st.markdown(f"**Calibration (top-label ECE = {ece:.4f}, 10 equal-width bins)**")
        show_df(fmt_table(bins, 3))
    if data.n_classes == 2:
        curve = reliability_curve_data(data.y_test, m.proba_test, data.positive_index)
        if curve is not None:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.plot([0, 1], [0, 1], ":", color=viz.MUTED)
            ax.plot(curve[0], curve[1], "o-", color=viz.PRIMARY)
            ax.set_xlabel("Mean predicted P(planet-like)")
            ax.set_ylabel("Observed fraction planet-like")
            ax.set_title("Reliability diagram", fontsize=10)
            show_fig(fig, f"reliability_{m.strategy}_{m.model_name}", f"{key}_rel")
