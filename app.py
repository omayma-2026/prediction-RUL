"""
app.py – RUL Monitor : suivi de la durée de vie restante de moteurs turbofan
NASA C-MAPSS FD001.  Lancer :  streamlit run app.py
Interface inspirée d'un cockpit / écran ECAM (thème sombre, voyants, radar de flotte).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="RUL Monitor · Aero", page_icon="✈️", layout="wide")

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
CYAN, BLUE = "#22d3ee", "#38bdf8"
GREEN, AMBER, RED = "#4ade80", "#fbbf24", "#f87171"
TEXT, MUTED, BG = "#e2e8f0", "#8aa0b8", "#050b18"
STATUS_COLOR = {"Critique": RED, "Attention": AMBER, "Normal": GREEN}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--panel:rgba(11,24,46,.85);--line:rgba(125,211,252,.22);--cyan:#22d3ee;--text:#f1f5f9;--muted:#bccade;}
html, body, [class*="css"], .stApp {font-family:'Inter','Segoe UI',system-ui,sans-serif;color:var(--text);font-size:17px;}
.stMarkdown p, [data-testid="stCaptionContainer"] p {font-size:1rem;line-height:1.55;}
.stApp{background:
    radial-gradient(1100px 560px at 85% -8%, rgba(14,116,144,.30), transparent 60%),
    linear-gradient(180deg,#060d1c,#08152b 55%,#060d1c);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(125,211,252,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(125,211,252,.03) 1px,transparent 1px);
  background-size:48px 48px;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#08152b,#060d1c);border-right:1px solid var(--line);}
.block-container{padding-top:1.1rem;max-width:1400px;position:relative;z-index:1;}

/* ---- Onglets ---- */
[role="tablist"]{gap:.5rem;border-bottom:1px solid var(--line);}
[role="tab"]{background:transparent;padding:.8rem 1.2rem;}
[role="tab"] p{font-family:'Inter',sans-serif;font-size:1.08rem;font-weight:600;letter-spacing:.01em;color:var(--muted)!important;}
[role="tab"][aria-selected="true"] p{color:#67e8f9!important;}

/* ---- Widgets ---- */
[data-testid="stWidgetLabel"] p{color:#d5e0ee!important;font-weight:600;font-size:.95rem;letter-spacing:.01em;}
.stDownloadButton button,[data-testid="stDownloadButton"] button{background:linear-gradient(90deg,#0e7490,#2563eb);
  color:#fff;border:0;font-family:'Inter',sans-serif;font-weight:600;font-size:1rem;border-radius:10px;padding:.7rem 1.4rem;}
.stDownloadButton button:hover,[data-testid="stDownloadButton"] button:hover{filter:brightness(1.12);color:#fff;}

/* ---- Panneaux ---- */
[data-testid="stVerticalBlock"]:has(> :is([data-testid="stElementContainer"],[data-testid="element-container"]) .ptitle){
  border:1px solid var(--line)!important;border-radius:14px!important;background:var(--panel)!important;
  box-shadow:0 8px 24px rgba(0,0,0,.28);padding:1rem 1.1rem .7rem!important;}

/* ---- En-tête ---- */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:20px;
  padding:1.7rem 280px 1.5rem 2rem;margin-bottom:1.1rem;
  background:linear-gradient(120deg,rgba(8,47,73,.85),rgba(8,21,43,.92) 55%,rgba(12,74,110,.5));}
.hero-title{font-family:'Orbitron',sans-serif;font-weight:900;font-size:2.4rem;letter-spacing:.08em;margin:0;color:#f8fafc;}
.hero-sub{font-family:'Inter',sans-serif;color:#d5e0ee;font-size:1.12rem;margin:.45rem 0 1rem;font-weight:500;}
.chip{display:inline-block;font-family:'Inter',sans-serif;font-size:.95rem;font-weight:500;
  padding:.35rem .9rem;margin:0 .45rem .4rem 0;border-radius:999px;border:1px solid var(--line);
  background:rgba(34,211,238,.09);color:#e0f2fe;}
.chip b{color:#67e8f9;font-weight:700;}
.engine-art{position:absolute;right:2.2rem;top:50%;transform:translateY(-50%);opacity:.95;}
.fan{transform-origin:110px 110px;animation:spin 9s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.route{position:absolute;inset:0;pointer-events:none;opacity:.5;}
@media (max-width:900px){.engine-art{display:none;}.hero{padding-right:1.4rem;}}

/* ---- Cartes KPI ---- */
.kpi{position:relative;border:1px solid var(--line);border-radius:14px;padding:1rem 1.2rem 1.05rem;
  background:var(--panel);box-shadow:0 8px 24px rgba(0,0,0,.28);height:100%;}
.kpi::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:4px;border-radius:4px;background:var(--c);}
.kpi-l{font-family:'Inter',sans-serif;font-size:1.02rem;font-weight:600;color:#d5e0ee;}
.kpi-v{font-family:'JetBrains Mono','Consolas',monospace;font-size:2.8rem;font-weight:700;line-height:1.2;color:var(--c);}
.kpi-u{font-family:'Inter',sans-serif;font-size:1.05rem;font-weight:500;color:var(--muted);margin-left:.45rem;}
.kpi-s{font-size:.98rem;color:var(--muted);}
.lamp{display:inline-block;width:16px;height:16px;border-radius:50%;background:var(--c);
  box-shadow:0 0 10px var(--c);margin-right:.65rem;vertical-align:middle;}
.status-txt{font-family:'Inter',sans-serif;font-weight:800;font-size:1.7rem;letter-spacing:.05em;color:var(--c);vertical-align:middle;}

/* ---- Titres de section / barres ---- */
.ptitle{font-family:'Inter',sans-serif;font-size:1.15rem;font-weight:700;color:#e8f4ff;
  margin:.1rem 0 .6rem;display:flex;align-items:center;gap:.65rem;}
.ptitle::before{content:"";width:5px;height:1.15em;border-radius:3px;background:var(--cyan);}
.health{height:10px;border-radius:10px;background:rgba(148,163,184,.18);overflow:hidden;margin:.5rem 0 .25rem;}
.health>div{height:100%;border-radius:10px;background:var(--c);}
.reco{border:1px solid var(--line);border-left:4px solid var(--c);border-radius:10px;padding:.9rem 1.1rem;
  background:rgba(8,20,40,.6);font-size:1.1rem;margin:.7rem 0 1rem;color:#f1f5f9;}
.reco b{font-weight:700;font-size:1.05rem;color:var(--c);}
.row{display:flex;align-items:center;gap:.8rem;padding:.5rem .2rem;border-bottom:1px solid rgba(148,163,184,.15);
  font-family:'JetBrains Mono','Consolas',monospace;font-size:1.02rem;}
.row .id{width:110px;color:#f1f5f9;} .row .val{width:56px;text-align:right;color:var(--c);font-weight:700;}
.row .bar{flex:1;height:7px;border-radius:7px;background:rgba(148,163,184,.18);overflow:hidden;}
.row .bar>div{height:100%;background:var(--c);}
.foot{color:var(--muted);font-size:.9rem;text-align:center;margin-top:1.5rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Aides d'affichage
# --------------------------------------------------------------------------- #
def html(s: str) -> str:
    """Aplatit un bloc HTML (évite que Markdown le prenne pour du code)."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def kpi(label, value, unit="", color=CYAN, sub=""):
    return html(
        f"""<div class="kpi" style="--c:{color}"><div class="kpi-l">{label}</div>
        <div class="kpi-v">{value}<span class="kpi-u">{unit}</span></div>
        <div class="kpi-s">{sub}&nbsp;</div></div>"""
    )


def ptitle(text):
    st.markdown(f'<div class="ptitle">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320, margin=None):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,40,.45)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#d5e0ee", size=15),
        margin=margin or dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=15)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.18)")
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, theme=None)
    except TypeError:
        st.plotly_chart(fig)


def hero_svg() -> str:
    """Turbofan (vue de face) animé + route de vol décorative."""
    blades = "".join(
        f'<path transform="rotate({i * 360 / 16:.2f} 110 110)" '
        f'd="M110 92 C121 76 126 52 116 28 L106 28 C103 52 104 76 110 92 Z" '
        f'fill="rgba(34,211,238,.16)" stroke="#22d3ee" stroke-width="1"/>'
        for i in range(16)
    )
    ticks = "".join(
        f'<line x1="110" y1="6" x2="110" y2="{14 if i % 5 else 20}" stroke="#38bdf8" stroke-opacity=".55" '
        f'stroke-width="{1 if i % 5 else 2}" transform="rotate({i * 6} 110 110)"/>'
        for i in range(60)
    )
    engine = (
        '<svg class="engine-art" width="220" height="220" viewBox="0 0 220 220">'
        f'{ticks}'
        '<circle cx="110" cy="110" r="96" fill="none" stroke="#22d3ee" stroke-opacity=".5" stroke-width="1.5"/>'
        '<circle cx="110" cy="110" r="90" fill="rgba(5,11,24,.85)" stroke="#0e7490" stroke-width="3"/>'
        f'<g class="fan">{blades}</g>'
        '<circle cx="110" cy="110" r="22" fill="#07142a" stroke="#22d3ee" stroke-width="2"/>'
        '<circle cx="110" cy="110" r="9" fill="#22d3ee" fill-opacity=".85"/>'
        "</svg>"
    )
    route = (
        '<svg class="route" viewBox="0 0 1000 200" preserveAspectRatio="none">'
        '<path d="M-20 170 C 220 20, 520 0, 760 90" fill="none" stroke="#22d3ee" stroke-opacity=".5" '
        'stroke-width="1.5" stroke-dasharray="3 7"/>'
        '<circle cx="180" cy="78" r="3" fill="#38bdf8"/><circle cx="430" cy="38" r="3" fill="#38bdf8"/>'
        '<circle cx="640" cy="52" r="3" fill="#38bdf8"/>'
        "</svg>"
    )
    return engine + route


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle (entraînement si nécessaire)…")
def get_model():
    return core.load_artifacts()


@st.cache_data(show_spinner="Calcul des variables glissantes…")
def get_features(_meta_key: str):
    return core.load_test_features(META)


MODEL, META = get_model()
FEAT, RUL_FINAL = get_features(META["model"])
FLEET = core.predict_fleet(MODEL, META, FEAT)
CAP = META["cap"]
RES = pd.DataFrame(META["results"]).T
BEST = META["model"]

# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
st.markdown(
    html(
        f"""<div class="hero">{hero_svg()}
        <div class="hero-title">RUL MONITOR</div>
        <div class="hero-sub">Suivi de la santé des moteurs d'avion · Maintenance prédictive</div>
        <span class="chip">Données : <b>NASA C-MAPSS FD001</b></span>
        <span class="chip">Flotte : <b>{FLEET.shape[0]} moteurs</b></span>
        <span class="chip">Modèle : <b>{BEST}</b></span>
        <span class="chip">Erreur moyenne (RMSE) : <b>{RES.loc[BEST, 'RMSE']:.1f} cycles</b></span>
        </div>"""
    ),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        html(
            f"""<div style="font-family:Inter,sans-serif;font-weight:700;color:{CYAN};font-size:1.05rem;
            margin:.4rem 0 1rem;"><span class="lamp" style="--c:{GREEN};width:10px;height:10px"></span>
            Panneau de commande</div>"""
        ),
        unsafe_allow_html=True,
    )
    unit = st.selectbox("Moteur (ID)", sorted(FEAT["unit"].unique()))
    eng = FEAT[FEAT["unit"] == unit].reset_index(drop=True)
    max_c = int(eng["cycle"].max())
    cycle = st.slider("Cycle courant (simulation du vol)", 1, max_c, max_c)
    st.divider()
    crit = st.number_input("Seuil critique (cycles)", 5, 100, core.CRIT_DEFAULT)
    warn = st.number_input("Seuil d'attention (cycles)", 10, 200, core.WARN_DEFAULT)
    st.caption("Sous le seuil critique : maintenance immédiate.")

tab_engine, tab_fleet, tab_models, tab_export = st.tabs(
    ["Suivi moteur", "Flotte", "Modèles", "Export"]
)

# --------------------------------------------------------------------------- #
# Onglet 1 : suivi d'un moteur
# --------------------------------------------------------------------------- #
with tab_engine:
    sub = eng[eng["cycle"] <= cycle]
    pred = pd.Series(core.predict_rul(MODEL, META, sub), index=sub["cycle"])
    rul_now = float(pred.iloc[-1])
    true_all = (RUL_FINAL[unit - 1] + (max_c - eng["cycle"])).clip(upper=CAP)
    true_now = float(true_all.iloc[cycle - 1])
    label, _ = core.status(rul_now, crit, warn)
    color = STATUS_COLOR[label]
    health = max(0.0, min(100.0, rul_now / CAP * 100))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("RUL prédite", f"{rul_now:.0f}", "cycles", color, f"Moteur {unit} · cycle {cycle}"),
                unsafe_allow_html=True)
    c2.markdown(kpi("RUL réelle", f"{true_now:.0f}", "cycles", TEXT, f"plafonnée à {CAP}"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Écart", f"{rul_now - true_now:+.0f}", "cycles", BLUE, "prédite − réelle"),
                unsafe_allow_html=True)
    c4.markdown(
        html(
            f"""<div class="kpi" style="--c:{color}"><div class="kpi-l">Statut du moteur</div>
            <div style="margin:.75rem 0 .5rem"><span class="lamp"></span><span class="status-txt">{label.upper()}</span></div>
            <div class="kpi-s">Santé estimée : {health:.0f} %</div></div>"""
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="health" style="--c:{color}"><div style="width:{health:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )
    reco = {
        "Critique": "Retirer le moteur du service et planifier une inspection immédiate.",
        "Attention": "Planifier une intervention de maintenance lors de la prochaine escale.",
        "Normal": "Fonctionnement nominal. Poursuivre la surveillance habituelle.",
    }[label]
    st.markdown(
        f'<div class="reco" style="--c:{color}"><b>Recommandation :</b>&nbsp;{reco}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])
    with left:
        with st.container(border=True):
            ptitle("Indicateur RUL")
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=rul_now,
                    number={"suffix": " cyc", "valueformat": ".0f", "font": {"family": "Inter, Segoe UI, sans-serif", "color": color, "size": 42}},
                    gauge={
                        "axis": {"range": [0, CAP], "tickcolor": "#64748b", "tickfont": {"color": "#cbd5e1", "size": 14}},
                        "bar": {"color": color, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, crit], "color": "rgba(239,68,68,.30)"},
                            {"range": [crit, warn], "color": "rgba(251,191,36,.26)"},
                            {"range": [warn, CAP], "color": "rgba(34,197,94,.20)"},
                        ],
                        "threshold": {"line": {"color": "#e2e8f0", "width": 3}, "thickness": 0.85, "value": rul_now},
                    },
                )
            )
            show(style_fig(gauge, 290, dict(l=25, r=25, t=25, b=5)))
    with right:
        with st.container(border=True):
            ptitle("Évolution de la RUL")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eng["cycle"][:cycle], y=true_all[:cycle], name="RUL réelle",
                                     line=dict(color="#e2e8f0", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="RUL prédite", fill="tozeroy",
                                     line=dict(color=CYAN, width=3), fillcolor="rgba(34,211,238,.10)"))
            fig.add_hline(y=crit, line_color=RED, line_dash="dot", annotation_text="critique",
                          annotation_font_color=RED, annotation_font_size=14)
            fig.add_hline(y=warn, line_color=AMBER, line_dash="dot", annotation_text="attention",
                          annotation_font_color=AMBER, annotation_font_size=14)
            fig.update_layout(xaxis_title="cycle", yaxis_title="RUL (cycles)", yaxis_range=[0, CAP + 8])
            show(style_fig(fig, 290))

    with st.container(border=True):
        ptitle("Capteurs")
        chosen = st.multiselect("Capteurs à afficher", META["sensors"], default=META["sensors"][:4])
        if chosen:
            long = sub.melt(id_vars="cycle", value_vars=chosen, var_name="capteur", value_name="valeur")
            fig_s = px.line(long, x="cycle", y="valeur", color="capteur", facet_row="capteur",
                            color_discrete_sequence=[CYAN, "#a78bfa", AMBER, GREEN, "#f472b6", BLUE])
            fig_s.update_yaxes(matches=None, title="")
            fig_s.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_color="#e2e8f0", font_size=15))
            fig_s.update_layout(showlegend=False)
            show(style_fig(fig_s, 165 * len(chosen) + 60, dict(l=10, r=10, t=20, b=10)))

# --------------------------------------------------------------------------- #
# Onglet 2 : flotte
# --------------------------------------------------------------------------- #
with tab_fleet:
    fleet = FLEET.copy()
    fleet["Statut"] = [core.status(v, crit, warn)[0] for v in fleet["RUL_predite"]]
    counts = fleet["Statut"].value_counts()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Moteurs suivis", len(fleet), "", CYAN, "flotte FD001"), unsafe_allow_html=True)
    k2.markdown(kpi("Critiques", int(counts.get("Critique", 0)), "", RED, f"RUL sous {crit} cycles"), unsafe_allow_html=True)
    k3.markdown(kpi("Attention", int(counts.get("Attention", 0)), "", AMBER, f"RUL sous {warn} cycles"), unsafe_allow_html=True)
    k4.markdown(kpi("Normaux", int(counts.get("Normal", 0)), "", GREEN, "fonctionnent normalement"), unsafe_allow_html=True)
    st.write("")

    rl, rr = st.columns([1.15, 1])
    with rl:
        with st.container(border=True):
            ptitle("Radar de la flotte (le centre = danger)")
            f = fleet.sort_values("moteur")
            theta = (f["moteur"] - 1) / f["moteur"].max() * 360
            ring_t = np.linspace(0, 360, 181)
            radar = go.Figure()
            for r_val, col in ((crit, RED), (warn, AMBER)):
                radar.add_trace(go.Scatterpolar(r=[r_val] * len(ring_t), theta=ring_t, mode="lines",
                                                line=dict(color=col, width=1.5, dash="dot"),
                                                hoverinfo="skip", showlegend=False))
            for name in ("Normal", "Attention", "Critique"):
                d = f[f["Statut"] == name]
                radar.add_trace(go.Scatterpolar(
                    r=d["RUL_predite"], theta=theta[d.index], mode="markers", name=name, text=d["moteur"],
                    marker=dict(size=10, color=STATUS_COLOR[name], line=dict(color=BG, width=1)),
                    hovertemplate="Moteur %{text}<br>RUL %{r:.0f} cycles<extra></extra>"))
            radar.update_layout(polar=dict(
                bgcolor="rgba(8,20,40,.55)",
                radialaxis=dict(range=[0, CAP], gridcolor="rgba(34,211,238,.18)", tickfont=dict(color="#94a3b8", size=12),
                                angle=90, tickangle=0),
                angularaxis=dict(gridcolor="rgba(34,211,238,.12)", showticklabels=False, rotation=90,
                                 direction="clockwise")))
            show(style_fig(radar, 430, dict(l=30, r=30, t=30, b=20)))
    with rr:
        with st.container(border=True):
            ptitle("Top 10 des moteurs à surveiller")
            rows = ""
            for _, r in fleet.sort_values("RUL_predite").head(10).iterrows():
                c = STATUS_COLOR[r["Statut"]]
                pct = max(2, min(100, r["RUL_predite"] / CAP * 100))
                rows += (f'<div class="row" style="--c:{c}"><span class="id">Moteur {int(r["moteur"]):03d}</span>'
                         f'<span class="bar"><div style="width:{pct:.0f}%"></div></span>'
                         f'<span class="val">{r["RUL_predite"]:.0f}</span></div>')
            st.markdown(html(rows), unsafe_allow_html=True)

    with st.container(border=True):
        ptitle("Tableau de la flotte")
        only = st.multiselect("Filtrer par statut", ["Critique", "Attention", "Normal"],
                              default=["Critique", "Attention", "Normal"])
        st.dataframe(
            fleet[fleet["Statut"].isin(only)].sort_values("RUL_predite"),
            hide_index=True,
            column_config={
                "RUL_predite": st.column_config.ProgressColumn("RUL prédite", min_value=0, max_value=CAP, format="%.0f"),
                "cycles_observes": "Cycles observés",
                "moteur": "Moteur",
            },
        )

# --------------------------------------------------------------------------- #
# Onglet 3 : comparaison des modèles
# --------------------------------------------------------------------------- #
with tab_models:
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Meilleur modèle", f'<span style="font-size:1.75rem">{BEST}</span>', "", CYAN, "plus faible erreur (RMSE)"), unsafe_allow_html=True)
    m2.markdown(kpi("RMSE", f"{RES.loc[BEST, 'RMSE']:.1f}", "cycles", BLUE, "erreur typique en cycles"), unsafe_allow_html=True)
    m3.markdown(kpi("MAE", f"{RES.loc[BEST, 'MAE']:.1f}", "cycles", GREEN, "écart moyen en cycles"), unsafe_allow_html=True)
    m4.markdown(kpi("R²", f"{RES.loc[BEST, 'R2']:.2f}", "", AMBER, "variance expliquée"), unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        ptitle("Performances sur le jeu de test (100 moteurs)")
        res = RES.round(2)
        st.dataframe(
            res.style.format("{:.2f}")
            .highlight_min(subset=["RMSE", "MAE", "NASA score"], props="background-color:rgba(34,211,238,.22);font-weight:700")
            .highlight_max(subset=["R2"], props="background-color:rgba(34,211,238,.22);font-weight:700")
        )
        st.caption("RMSE / MAE en cycles · score NASA : plus bas = mieux (pénalise les prédictions trop optimistes).")

    cl, cr = st.columns(2)
    with cl:
        with st.container(border=True):
            ptitle("Erreurs par modèle")
            m = res[["RMSE", "MAE"]].reset_index().melt(id_vars="index", var_name="métrique", value_name="cycles")
            fig_m = px.bar(m, x="index", y="cycles", color="métrique", barmode="group",
                           labels={"index": "modèle"}, color_discrete_sequence=[CYAN, "#a78bfa"])
            show(style_fig(fig_m, 340))
    with cr:
        with st.container(border=True):
            ptitle("RUL prédite vs RUL réelle")
            preds = pd.DataFrame(META["preds"])
            names = list(META["results"].keys())
            name = st.selectbox("Modèle", names, index=names.index(BEST))
            sc = px.scatter(preds, x="RUL_reelle", y=f"RUL_predite_{name}", hover_data=["moteur"],
                            labels={"RUL_reelle": "RUL réelle", f"RUL_predite_{name}": "RUL prédite"},
                            color_discrete_sequence=[CYAN])
            sc.add_shape(type="line", x0=0, y0=0, x1=CAP, y1=CAP, line=dict(color=RED, dash="dash"))
            show(style_fig(sc, 285))

    with st.container(border=True):
        ptitle("Les 15 variables les plus importantes (Random Forest)")
        imp = pd.DataFrame(META["importance"], columns=["variable", "importance"]).head(15).iloc[::-1]
        fig_i = px.bar(imp, x="importance", y="variable", orientation="h", color_discrete_sequence=[CYAN])
        show(style_fig(fig_i, 440))

# --------------------------------------------------------------------------- #
# Onglet 4 : export
# --------------------------------------------------------------------------- #
with tab_export:
    with st.container(border=True):
        ptitle("Télécharger les résultats")
        st.write("Le rapport Excel contient 3 feuilles : **Métriques**, **Prédictions** (par moteur) "
                 "et **Importance variables**.")
        e1, e2 = st.columns(2)
        e1.download_button(
            "Télécharger le rapport Excel",
            data=core.build_excel(META),
            file_name="resultats_RUL_FD001.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        fleet_csv = FLEET.assign(Statut=[core.status(v, crit, warn)[0] for v in FLEET["RUL_predite"]])
        e2.download_button(
            "Télécharger l'état de la flotte (CSV)",
            data=fleet_csv.to_csv(index=False).encode("utf-8"),
            file_name="etat_flotte.csv",
            mime="text/csv",
        )

st.markdown('<div class="foot">RUL Monitor · NASA C-MAPSS FD001 · Projet de maintenance prédictive</div>',
            unsafe_allow_html=True)
