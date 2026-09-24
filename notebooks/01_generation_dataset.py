"""
Corridor Risk Intelligence — Génération du dataset simulé
Calibré sur des statistiques réelles publiées sur le Port Autonome de Dakar (PAD)
et le corridor Dakar-Bamako (voir README pour les sources).

Hypothèses de calibrage (sourcées) :
- Croissance des volumes conteneurs à l'import : +10 à 12%/an -> pression continue
  sur la capacité des terminaux, donc congestion croissante au fil du temps simulé.
- Délai moyen de retour du conteneur vide : ~40h en moyenne (peut grimper fortement
  en période de forte congestion).
- Plus de 1000 camions/jour sur le corridor Dakar-Bamako.
- Le PAD concentre plus de 70% des importations du pays.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N = 6000  # nombre d'expéditions simulées

# --- Période simulée : 18 mois, avec croissance de volume (+11%/an en moyenne) ---
start_date = datetime(2024, 1, 1)
date_range_days = 540
dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, date_range_days, N)]
dates.sort()

terminals = ["Terminal à Conteneurs", "Môle 1", "Môle 3", "Môle 8", "Terminal Roulier"]
terminal_weights = [0.38, 0.18, 0.16, 0.14, 0.14]

merchandise_types = ["Import général", "Denrées alimentaires", "Matériaux de construction",
                      "Hydrocarbures", "Biens de consommation", "Équipements industriels"]
merchandise_weights = [0.32, 0.20, 0.16, 0.10, 0.14, 0.08]

destinations = {
    "Dakar (local)": 20, "Thiès": 70, "Kaolack": 190, "Touba": 190,
    "Tambacounda": 470, "Kaédi (Mauritanie)": 470, "Bamako (Mali)": 1220,
    "Kayes (Mali)": 950, "Ziguinchor": 450,
}
dest_names = list(destinations.keys())
dest_weights = [0.22, 0.10, 0.10, 0.08, 0.09, 0.05, 0.20, 0.10, 0.06]

# Transporteurs génériques et anonymisés (aucun lien avec une entreprise réelle) :
# le score de fiabilité est une hypothèse de modélisation arbitraire, à remplacer
# par un historique réel en production — voir README pour cette précision.
carriers = ["Transporteur A", "Transporteur B", "Transporteur C", "Transporteur D",
            "Transporteur E", "Transporteur F", "Transporteur G", "Transporteur indépendant"]
carrier_reliability = {  # score de fiabilité SIMULÉ (0 = peu fiable, 1 = très fiable)
    "Transporteur A": 0.80, "Transporteur B": 0.85, "Transporteur C": 0.60,
    "Transporteur D": 0.55, "Transporteur E": 0.50, "Transporteur F": 0.65,
    "Transporteur G": 0.70, "Transporteur indépendant": 0.35,
}

rows = []
for i, d in enumerate(dates):
    # --- Progression temporelle de la congestion (+11%/an ~ +0.030%/jour composés) ---
    day_index = (d - start_date).days
    growth_factor = (1 + 0.11) ** (day_index / 365)

    # Volume journalier du terminal (proxy congestion), avec bruit + saisonnalité hebdo
    weekday = d.weekday()  # 0=lundi
    weekday_factor = 1.15 if weekday in (0, 1) else (0.75 if weekday == 6 else 1.0)
    base_daily_volume = 340 * growth_factor * weekday_factor
    volume_journalier_terminal = max(50, int(rng.normal(base_daily_volume, base_daily_volume * 0.12)))

    terminal = rng.choice(terminals, p=terminal_weights)
    merchandise = rng.choice(merchandise_types, p=merchandise_weights)
    volume_evp = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.10, 0.05]))

    dest = rng.choice(dest_names, p=dest_weights)
    distance_km = destinations[dest] * rng.normal(1.0, 0.03)

    carrier = rng.choice(carriers)
    reliability = carrier_reliability[carrier]

    month = d.month
    saison = "pluies" if month in (6, 7, 8, 9, 10) else "sèche"

    statut_douane = rng.choice(
        ["dédouané avant arrivée", "dédouanement en cours", "bloqué (litige/documents)"],
        p=[0.45, 0.42, 0.13]
    )

    # --- Cible 1 : temps de séjour au port (heures) ---
    congestion_pressure = (volume_journalier_terminal / 300) ** 1.6
    douane_penalty = {"dédouané avant arrivée": 0, "dédouanement en cours": 18,
                       "bloqué (litige/documents)": 60}[statut_douane]
    base_dwell = 30 + 22 * congestion_pressure + douane_penalty
    # shape plus élevé = moins de bruit relatif -> le signal métier (douane, congestion)
    # reste détectable par un modèle, au lieu d'être noyé dans l'aléatoire
    temps_sejour_port_heures = max(6, rng.gamma(shape=12.0, scale=base_dwell / 12.0))

    # --- Cible 2 : retour du conteneur vide (base ~40h, calibré article Africa Supply Chain) ---
    base_retour_vide = 40 * (1 + 0.5 * (congestion_pressure - 1))
    temps_retour_vide_heures = max(8, rng.gamma(shape=9.0, scale=base_retour_vide / 9.0))

    # --- Cible 3 : retard sur le corridor routier (bool) ---
    saison_penalty = 0.18 if saison == "pluies" else 0.0
    distance_penalty = min(0.30, distance_km / 4000)
    risk_score = (0.30 * congestion_pressure / 2 + 0.30 * (1 - reliability)
                  + saison_penalty + distance_penalty + rng.normal(0, 0.08))
    retard_corridor = bool(risk_score > 0.55)

    # --- Coût de détention estimé (FCFA), proportionnel au dépassement de délai ---
    seuil_gratuit_heures = 72  # franchise standard avant facturation détention
    depassement = max(0, (temps_sejour_port_heures + temps_retour_vide_heures) - seuil_gratuit_heures)
    cout_detention_estime_fcfa = int(depassement * rng.uniform(4500, 7000) * volume_evp)

    rows.append({
        "shipment_id": f"CRI-{i+1:05d}",
        "date_arrivee_port": d.strftime("%Y-%m-%d"),
        "terminal": terminal,
        "type_marchandise": merchandise,
        "volume_evp": volume_evp,
        "volume_journalier_terminal": volume_journalier_terminal,
        "compagnie_transport": carrier,
        "destination_finale": dest,
        "distance_corridor_km": round(distance_km, 1),
        "jour_semaine": ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][weekday],
        "saison": saison,
        "statut_douane": statut_douane,
        "temps_sejour_port_heures": round(temps_sejour_port_heures, 1),
        "temps_retour_conteneur_vide_heures": round(temps_retour_vide_heures, 1),
        "retard_corridor": retard_corridor,
        "cout_detention_estime_fcfa": cout_detention_estime_fcfa,
    })

df = pd.DataFrame(rows)
df.to_csv("/mnt/user-data/outputs/corridor_risk_dataset.csv", index=False)

print(df.shape)
print(df.head(10).to_string())
print("\n--- Stats clés ---")
print("Temps de séjour port (h): moyenne", round(df.temps_sejour_port_heures.mean(),1),
      "| médiane", round(df.temps_sejour_port_heures.median(),1))
print("Retour conteneur vide (h): moyenne", round(df.temps_retour_conteneur_vide_heures.mean(),1))
print("Taux de retard corridor:", round(df.retard_corridor.mean()*100,1), "%")
print("Coût détention moyen (FCFA):", int(df.cout_detention_estime_fcfa.mean()))
print("Coût détention total simulé (FCFA):", int(df.cout_detention_estime_fcfa.sum()))
