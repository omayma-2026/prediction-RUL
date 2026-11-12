"""
app.py – Interface Streamlit de suivi de la durée de vie restante (RUL)
NASA C-MAPSS FD001.  Lancer :  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="RUL Monitor · C-MAPSS", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1400px;}
    .hero {background: linear-gradient(135deg, #0b3d91 0%, #1565c0 55%, #00838f 100%);
           padding: 1.5rem 2rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 2rem; color: #fff; font-weight: 700;}
    .hero p {margin: .35rem 0 0; opacity: .92; font-size: 1.02rem;}
    .badge {display: inline-block; padding: .45rem 1.1rem; border-radius: 999px;
            font-weight: 700; color: #fff; font-size: 1.05rem;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


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

# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero">
      <h1>✈️ RUL Monitor – Maintenance prédictive</h1>
      <p>Prédiction de la durée de vie restante de moteurs turbofan · NASA C-MAPSS (FD001) ·
         modèle actif : <b>{META['model']}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Paramètres")
    unit = st.selectbox("Moteur", sorted(FEAT["unit"].unique()))
    eng = FEAT[FEAT["unit"] == unit].reset_index(drop=True)
    max_c = int(eng["cycle"].max())
    cycle = st.slider("Cycle courant (simulation du suivi)", 1, max_c, max_c)
    st.divider()
    crit = st.number_input("Seuil critique (cycles)", 5, 100, core.CRIT_DEFAULT)
    warn = st.number_input("Seuil d'attention (cycles)", 10, 200, core.WARN_DEFAULT)
    st.caption("En dessous du seuil critique : maintenance immédiate.")

tab_engine, tab_fleet, tab_models, tab_export = st.tabs(
    ["🔧 Suivi d'un moteur", "🛩️ Vue flotte", "📊 Comparaison des modèles", "📥 Export Excel"]
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
    label, color = core.status(rul_now, crit, warn)

    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("RUL prédite", f"{rul_now:.0f} cycles")
    with c2.container(border=True):
        st.metric("RUL réelle (plafonnée)", f"{true_now:.0f} cycles")
    with c3.container(border=True):
        st.metric("Erreur", f"{rul_now - true_now:+.0f} cycles")
    with c4.container(border=True):
        st.markdown("**Statut**")
        st.markdown(f'<span class="badge" style="background:{color}">{label}</span>', unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rul_now,
                number={"suffix": " cycles"},
                gauge={
                    "axis": {"range": [0, CAP]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, crit], "color": "rgba(211,47,47,.25)"},
                        {"range": [crit, warn], "color": "rgba(245,124,0,.25)"},
                        {"range": [warn, CAP], "color": "rgba(46,125,50,.20)"},
                    ],
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(gauge)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="RUL prédite", line=dict(width=3)))
        fig.add_trace(
            go.Scatter(x=eng["cycle"][:cycle], y=true_all[:cycle], name="RUL réelle", line=dict(dash="dash"))
        )
        fig.add_hline(y=crit, line_color="#d32f2f", line_dash="dot", annotation_text="critique")
        fig.add_hline(y=warn, line_color="#f57c00", line_dash="dot", annotation_text="attention")
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="cycle", yaxis_title="RUL (cycles)", yaxis_range=[0, CAP + 5],
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig)

    st.subheader("Capteurs")
    default = META["sensors"][:4]
    chosen = st.multiselect("Capteurs à afficher", META["sensors"], default=default)
    if chosen:
        long = sub.melt(id_vars="cycle", value_vars=chosen, var_name="capteur", value_name="valeur")
        fig_s = px.line(long, x="cycle", y="valeur", color="capteur", facet_row="capteur")
        fig_s.update_yaxes(matches=None, title="")
        fig_s.update_layout(height=170 * len(chosen) + 60, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        fig_s.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_s)

# --------------------------------------------------------------------------- #
# Onglet 2 : vue flotte
# --------------------------------------------------------------------------- #
with tab_fleet:
    fleet = FLEET.copy()
    fleet["Statut"] = [core.status(v, crit, warn)[0] for v in fleet["RUL_predite"]]
    counts = fleet["Statut"].value_counts()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Moteurs suivis", len(fleet))
    k2.metric("🔴 Critiques", int(counts.get("Critique", 0)))
    k3.metric("🟠 Attention", int(counts.get("Attention", 0)))
    k4.metric("🟢 Normaux", int(counts.get("Normal", 0)))

    order = fleet.sort_values("RUL_predite")
    fig_f = px.bar(
        order, x=order["moteur"].astype(str), y="RUL_predite", color="Statut",
        color_discrete_map={"Critique": "#d32f2f", "Attention": "#f57c00", "Normal": "#2e7d32"},
        labels={"x": "moteur", "RUL_predite": "RUL prédite (cycles)"},
    )
    fig_f.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis={"categoryorder": "array",
                        "categoryarray": order["moteur"].astype(str).tolist()})
    st.plotly_chart(fig_f)

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
    results = pd.DataFrame(META["results"]).T.round(2)
    st.subheader("Performances sur le jeu de test (100 moteurs)")
    st.dataframe(results.style.highlight_min(subset=["RMSE", "MAE", "NASA score"], color="#c8e6c9")
                 .highlight_max(subset=["R2"], color="#c8e6c9"))
    st.caption("RMSE / MAE en cycles · score NASA : plus bas = mieux (pénalise les prédictions trop optimistes).")

    m = results[["RMSE", "MAE"]].reset_index().melt(id_vars="index", var_name="métrique", value_name="cycles")
    st.plotly_chart(
        px.bar(m, x="index", y="cycles", color="métrique", barmode="group", labels={"index": "modèle"})
        .update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10)),
    )

    preds = pd.DataFrame(META["preds"])
    name = st.selectbox("Prédit vs réel pour le modèle :", list(META["results"].keys()),
                        index=list(META["results"].keys()).index(META["model"]))
    sc = px.scatter(preds, x="RUL_reelle", y=f"RUL_predite_{name}", hover_data=["moteur"],
                    labels={"RUL_reelle": "RUL réelle", f"RUL_predite_{name}": "RUL prédite"})
    sc.add_shape(type="line", x0=0, y0=0, x1=CAP, y1=CAP, line=dict(color="red", dash="dash"))
    sc.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(sc)

    imp = pd.DataFrame(META["importance"], columns=["variable", "importance"]).head(15).iloc[::-1]
    st.plotly_chart(
        px.bar(imp, x="importance", y="variable", orientation="h", title="Top 15 variables (Random Forest)")
        .update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10)),
    )

# --------------------------------------------------------------------------- #
# Onglet 4 : export Excel
# --------------------------------------------------------------------------- #
with tab_export:
    st.subheader("Télécharger les résultats")
    st.write("Le rapport Excel contient 3 feuilles : **Métriques**, **Prédictions** (par moteur) "
             "et **Importance variables**.")
    st.download_button(
        "📥 Télécharger le rapport Excel",
        data=core.build_excel(META),
        file_name="resultats_RUL_FD001.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    fleet_csv = FLEET.assign(Statut=[core.status(v, crit, warn)[0] for v in FLEET["RUL_predite"]])
    st.download_button(
        "📄 Télécharger l'état de la flotte (CSV)",
        data=fleet_csv.to_csv(index=False).encode("utf-8"),
        file_name="etat_flotte.csv",
        mime="text/csv",
    )
