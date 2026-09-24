import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Corridor Risk Intelligence", layout="wide")

# ============================================================
# Chargement des modèles et métadonnées
# ============================================================
@st.cache_resource
def load_artifacts():
    clf_retard = joblib.load("modele_retard_corridor.pkl")
    clf_franchise = joblib.load("modele_depassement_franchise.pkl")
    meta = joblib.load("feature_meta.pkl")
    return clf_retard, clf_franchise, meta

clf_retard, clf_franchise, meta = load_artifacts()
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]
options = meta["options"]

st.title("🚢 Corridor Risk Intelligence")
st.caption(
    "Simulateur de risque logistique — Port Autonome de Dakar → corridor Dakar-Bamako. "
    "Modèles entraînés sur un dataset simulé, calibré sur des statistiques publiques du secteur "
    "(voir README pour les sources et les limites méthodologiques)."
)

# ============================================================
# Formulaire de saisie — une expédition
# ============================================================
st.sidebar.header("Caractéristiques de l'expédition")

terminal = st.sidebar.selectbox("Terminal", options["terminal"])
type_marchandise = st.sidebar.selectbox("Type de marchandise", options["type_marchandise"])
compagnie_transport = st.sidebar.selectbox("Transporteur", options["compagnie_transport"])
destination_finale = st.sidebar.selectbox("Destination finale", options["destination_finale"])
jour_semaine = st.sidebar.selectbox("Jour d'arrivée", options["jour_semaine"])
saison = st.sidebar.selectbox("Saison", options["saison"])
statut_douane_options = options["statut_douane"]
default_statut_idx = (
    statut_douane_options.index("dédouanement en cours")
    if "dédouanement en cours" in statut_douane_options else 0
)
statut_douane = st.sidebar.selectbox(
    "Statut douanier", statut_douane_options, index=default_statut_idx)

volume_evp = st.sidebar.slider("Volume (EVP)", 1, 4, 1)
volume_journalier_terminal = st.sidebar.slider(
    "Volume journalier du terminal (nb conteneurs traités ce jour)", 50, 600, 370)
distance_map = {
    "Dakar (local)": 20, "Thiès": 70, "Kaolack": 190, "Touba": 190,
    "Tambacounda": 470, "Kaédi (Mauritanie)": 470, "Bamako (Mali)": 1220,
    "Kayes (Mali)": 950, "Ziguinchor": 450,
}
distance_corridor_km = distance_map.get(destination_finale, 200)
mois = st.sidebar.slider("Mois d'arrivée", 1, 12, 6)

# ============================================================
# Préparation des features (identique au pipeline d'entraînement)
# ============================================================
mois_sin = np.sin(2 * np.pi * mois / 12)
mois_cos = np.cos(2 * np.pi * mois / 12)
douane_bloque = int(statut_douane == "bloqué (litige/documents)")
douane_en_cours = int(statut_douane == "dédouanement en cours")
congestion_x_douane_bloque = volume_journalier_terminal * douane_bloque
congestion_sq = volume_journalier_terminal ** 1.6
volume_journalier_log = np.log1p(volume_journalier_terminal)

row = pd.DataFrame([{
    "terminal": terminal, "type_marchandise": type_marchandise,
    "compagnie_transport": compagnie_transport, "destination_finale": destination_finale,
    "jour_semaine": jour_semaine, "saison": saison, "statut_douane": statut_douane,
    "volume_evp": volume_evp, "volume_journalier_terminal": volume_journalier_terminal,
    "distance_corridor_km": distance_corridor_km, "mois_sin": mois_sin, "mois_cos": mois_cos,
    "douane_bloque": douane_bloque, "douane_en_cours": douane_en_cours,
    "congestion_x_douane_bloque": congestion_x_douane_bloque,
    "congestion_sq": congestion_sq, "volume_journalier_log": volume_journalier_log,
}])

# ============================================================
# Prédictions
# ============================================================
proba_retard = clf_retard.predict_proba(row)[0, 1]
proba_franchise = clf_franchise.predict_proba(row)[0, 1]

col1, col2 = st.columns(2)
with col1:
    st.metric("Risque de retard corridor (Dakar→destination)", f"{proba_retard*100:.0f}%")
with col2:
    st.metric("Risque de dépassement franchise douane (>72h)", f"{proba_franchise*100:.0f}%")

# ============================================================
# Estimation du coût de détention évitable
# ============================================================
st.subheader("💰 Impact financier estimé")
cout_unitaire_moyen_fcfa = 5750  # milieu de la fourchette utilisée à la génération du dataset
depassement_estime_h = max(0, 40 * proba_franchise * 2)  # proxy simple, transparent
cout_estime = int(depassement_estime_h * cout_unitaire_moyen_fcfa * volume_evp)
st.write(
    f"Si cette expédition dépasse la franchise, le surcoût de détention estimé est de "
    f"**~{cout_estime:,} FCFA**".replace(",", " ")
)
st.caption("Estimation indicative basée sur les mêmes hypothèses de coût que le dataset simulé — à calibrer sur des tarifs réels en production.")

# ============================================================
# Explicabilité SHAP
# ============================================================
st.subheader("🔍 Pourquoi ce score ? (SHAP)")

def explain(pipeline, row, label):
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    row_transformed = prep.transform(row)
    feature_names = prep.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_transformed)
    contrib = pd.Series(shap_values[0], index=feature_names)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    st.write(f"**{label}** — principaux facteurs :")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact SHAP (+ augmente le risque / - le réduit)")
    st.pyplot(fig)

tab1, tab2 = st.tabs(["Retard corridor", "Dépassement franchise"])
with tab1:
    explain(clf_retard, row, "Retard corridor")
with tab2:
    explain(clf_franchise, row, "Dépassement franchise")

# ============================================================
# Recommandation d'action
# ============================================================
st.subheader("✅ Recommandation")
if proba_retard >= 0.6 or proba_franchise >= 0.6:
    st.error(
        "**Risque élevé** — Prioriser cette expédition : accélérer le dédouanement, "
        "confirmer le transporteur à l'avance, envisager un transporteur alternatif si "
        "le statut douanier reste bloqué."
    )
elif proba_retard >= 0.35 or proba_franchise >= 0.35:
    st.warning(
        "**Risque modéré** — Surveiller l'expédition, relance préventive du transporteur "
        "et vérification du statut douanier recommandées."
    )
else:
    st.success("**Risque faible** — Traitement standard, aucune action particulière requise.")

st.divider()
st.caption(
    "Projet de portfolio data science — dataset simulé et calibré sur des statistiques "
    "publiques du secteur logistique sénégalais. Voir le README du dépôt pour les sources, "
    "les hypothèses de modélisation et les limites (R² du modèle de temps de séjour, "
    "transporteurs anonymisés, etc.)."
)
