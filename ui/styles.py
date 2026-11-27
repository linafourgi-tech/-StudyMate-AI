"""Local visual theme. Native Streamlit widgets retain their keyboard behavior."""

from base64 import b64encode
from functools import lru_cache
from pathlib import Path

import streamlit as st


@lru_cache(maxsize=1)
def _desk_image() -> str:
    image = Path(__file__).with_name("assets") / "study-desk.webp"
    return b64encode(image.read_bytes()).decode("ascii")


CSS = """
<style>
:root { --ink:#19352d; --muted:#52665b; --rule:#d7dfd5; --pine:#18553e;
        --serif:"STIX Two Text","Source Serif Pro",Georgia,serif; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 2.4rem clamp(1.1rem,4vw,4rem) 4rem; }
[data-testid="stHeader"] { background:transparent; pointer-events:none; }
[data-testid="stHeader"] button, [data-testid="stHeader"] a { pointer-events:auto; }
[data-testid="stSidebar"] { background:#f2f3ed; border-right:1px solid var(--rule); }
[data-testid="stSidebarUserContent"] { padding-top:.1rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.38rem; }
.sm-brand { display:flex; align-items:center; gap:.68rem; margin:0 0 2.2rem; color:var(--ink); }
.sm-brand b { font:600 1.65rem/1 var(--serif); letter-spacing:-.07em; }
.sm-brand small { font: bold .73rem/1 sans-serif; color:var(--pine); letter-spacing:.06em; }
.sm-mark { display:inline-grid; place-items:center; width:40px; height:40px; border-radius:12px;
           background:var(--pine); color:#fff6da; font:italic bold 1.65rem/1 Georgia,serif; }
.sm-label { font-size:.74rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
            color:var(--muted); margin:0 0 .38rem; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button { justify-content:flex-start;
    text-align:left; border-radius:.65rem; border-color:transparent; min-height:2.65rem; padding-left:.85rem; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] {
    background:#dce9dc; color:#124531; border-color:#c9ddce; box-shadow:none; }
.sm-side { border-top:1px solid #d1dbcf; padding-top:1.2rem; margin-top:1.5rem; }
.sm-side-name { overflow-wrap:anywhere; font-weight:600; line-height:1.35; margin-bottom:1rem; }
.sm-hero { border-radius:1.3rem; padding:clamp(1.7rem,4vw,3.5rem); min-height:210px;
    display:flex; align-items:center; background-color:#142f24; background-size:cover;
    background-position:center 57%; color:#fffdf2;
    --hero-overlay:linear-gradient(90deg,#0b221bf0 0%,#102a22d2 43%,#102a2230 70%,transparent 85%); }
.sm-hero h1, .sm-title h1 { font:500 clamp(2.4rem,4.3vw,4.1rem)/1.05 var(--serif);
    letter-spacing:-.055em; margin:0 0 .7rem; padding:0; }
.sm-hero h1 { color:#fffdf2; }
.sm-hero p { color:#f0f4e8; max-width:32ch; font-size:1.08rem; line-height:1.45; margin:0; }
.sm-title { border-bottom:1px solid var(--rule); padding:.45rem 0 1.15rem; margin-bottom:.3rem; }
.sm-title h1 { color:var(--ink); }
.sm-title p { color:var(--muted); font-size:1.04rem; margin:0; line-height:1.5; overflow-wrap:anywhere; }
.sm-h2 { font:600 clamp(1.45rem,2vw,1.95rem)/1.2 var(--serif); color:var(--ink);
    letter-spacing:-.035em; margin:.1rem 0 .35rem; padding:0; overflow-wrap:anywhere; }
.sm-copy, .sm-passage { color:#30483c; line-height:1.6; margin:.2rem 0 .4rem; overflow-wrap:anywhere; }
.sm-small { color:var(--muted); font-size:.9rem; margin:.2rem 0; overflow-wrap:anywhere; }
.sm-passage { border-left:3px solid #9bb99f; padding-left:.9rem; white-space:pre-line; }
.sm-badge { display:inline-block; padding:.18rem .65rem; border-radius:1rem; background:#e4e9e3;
    color:#344d42; font-size:.82rem; font-weight:650; line-height:1.5; }
.sm-easy,.sm-correct { background:#dfefe2; color:#185031; }
.sm-medium,.sm-partial { background:#f5e8c2; color:#694807; }
.sm-hard,.sm-incorrect { background:#f7ddd7; color:#883523; }
.sm-question { font:500 clamp(1.4rem,2.4vw,2rem)/1.28 var(--serif); margin:.35rem 0;
    color:var(--ink); letter-spacing:-.025em; overflow-wrap:anywhere; }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"][data-border="true"] {
    border-radius:1rem; background:#fffefa; box-shadow:0 3px 15px #173d2808; }
[data-testid="stMetric"] { padding:.8rem 1rem; border:1px solid var(--rule); border-radius:.75rem;
    background:#fffefa; height:100%; }
[data-testid="stMetricValue"] { font-family:var(--serif); color:var(--ink); }
[data-testid="stFileUploaderDropzone"] { background:#f3f6ee; border:1px dashed #9bb39e; border-radius:.75rem; }
[data-testid="stRadio"] label { overflow-wrap:anywhere; }
[data-testid="stSidebar"] button:focus-visible, button:focus-visible { outline:3px solid #ae7b2d; }
@media (max-width:760px) {
    [data-testid="stMainBlockContainer"] { padding-top:3.4rem; }
    .sm-hero { min-height:175px; background-position:43% center;
        --hero-overlay:linear-gradient(90deg,#0b211bf5 5%,#102a22cd 54%,#102a22a8 100%); }
    .sm-hero p { max-width:27ch; }
}
</style>
"""


def apply_theme() -> None:
    st.html(CSS)


def lecture_header() -> None:
    background = f"var(--hero-overlay), url(data:image/webp;base64,{_desk_image()})"
    st.html(
        f'<section class="sm-hero" style="background-image:{background}"><div>'
        '<h1>Lecture workspace</h1><p>Turn your lecture into a cited study guide '
        'and a practice quiz.</p></div></section>'
    )
