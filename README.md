# 🚢 Corridor Risk Intelligence

Anticipation des retards logistiques et des surcoûts de détention sur la chaîne
**Port Autonome de Dakar → corridor routier Dakar-Bamako**.

Projet de portfolio data science visant à démontrer une approche de machine
learning appliquée à un problème métier réel du secteur logistique sénégalais.

---

## 1. Contexte business

Le Port Autonome de Dakar (PAD) est le principal point d'entrée des
marchandises importées au Sénégal, et fait face à une pression croissante :

- Les volumes conteneurs à l'import augmentent en moyenne de **10 à 12 % par an**,
  générant une congestion structurelle des terminaux
  ([Jeune Afrique, avril 2024](https://www.jeuneafrique.com/1557580/economie-entreprises/au-senegal-le-port-autonome-de-dakar-navigue-toujours-en-eaux-troubles/)).
- Les transporteurs routiers rencontrent régulièrement des difficultés à
  retourner un conteneur vide au terminal, avec un **retard moyen d'environ
  40 heures**, ce qui déclenche des **frais de détention automatiques**
  facturés par les armateurs
  ([Africa Supply Chain Magazine](https://www.africasupplychainmag.com/analyse-de-la-problematique-du-trafic-conteneurise-au-port-de-dakar/)).
- Plus de **1 000 camions par jour** circulent sur le corridor Dakar-Bamako,
  contribuant à l'engorgement des terminaux
  ([Jeune Afrique, 2020](https://www.jeuneafrique.com/mag/1077773/economie-entreprises/senegal-le-port-de-dakar-futur-hub-logistique-pour-lafrique-de-louest/)).
- Ces difficultés de délais et de congestion restent d'actualité en 2026
  ([Le Nouveau Manager, juin 2026](https://www.lenouveaumanager.info/port-autonome-de-dakar-les-defis-dune-plateforme-strategique-pour-lafrique-de-louest/)).
- Les acteurs majeurs du secteur, comme AGL, misent sur la digitalisation et
  des corridors plus fluides et plus intégrés comme axe de compétitivité
  ([Financial Afrik, juin 2026](https://www.financialafrik.com/2026/06/16/mohamed-diop-la-competitivite-logistique-de-lafrique-passera-par-des-corridors-plus-fluides-plus-digitaux-et-plus-integres/)).

**Le fil conducteur du problème** : congestion portuaire → retour tardif des
conteneurs vides → frais de détention → corridor routier saturé. Ce projet
explore comment un système de scoring de risque, explicable et actionnable,
peut aider à anticiper ces retards en amont plutôt que de les subir.

---

## 2. Transparence méthodologique

**Point important, à lire avant d'interpréter les résultats.**

PAD, AGL et les transporteurs sénégalais ne publient pas leurs données
opérationnelles : il n'existe pas de dataset ouvert sur la congestion réelle
du port ou les délais réels par expédition. Plutôt que d'utiliser un dataset
générique sans rapport avec le Sénégal (type e-commerce Brésil), ce projet
construit un **dataset simulé, mais calibré sur les statistiques réelles
listées en section 1** (croissance des volumes, délai moyen de retour à vide,
volume de camions/jour).

Ce que cela implique concrètement :

- Les valeurs numériques de congestion, saisonnalité et délai de retour à
  vide suivent des distributions calées sur les ordres de grandeur publiés.
- Les **transporteurs sont anonymisés et génériques** (Transporteur A, B, C...)
 aucun score de fiabilité n'est associé à une entreprise réelle. Ces scores
  sont une hypothèse de modélisation, à remplacer par un historique réel en
  production.
- Le modèle de **régression du temps de séjour en heures exactes** a été
  testé mais écarté comme livrable principal : son R² plafonne à **0,42**
  après feature engineering, ce qui est cohérent avec la littérature sur la
  prédiction de dwell time portuaire (généralement R² 0,3-0,7, car de
  nombreux facteurs réels grèves, pannes, aléas administratifs restent
  imprévisibles), mais reste insuffisant pour une mise en avant en heures
  précises.
- Les **deux modèles de classification** (section 3) sont donc les livrables
  retenus : ils répondent à une question métier plus actionnable ("ce
  conteneur est-il à risque, oui ou non ?") et atteignent une fiabilité
  nettement meilleure.

Ce projet doit être lu comme une **preuve de méthodologie transférable** :
l'approche (feature engineering, modélisation, explicabilité, interface de
décision) est directement réutilisable sur des données réelles PAD/AGL, si
un accès à ces données venait à être accordé.

---

## 3. Résultats des modèles retenus

Deux modèles de classification XGBoost, entraînés sur un pipeline
`ColumnTransformer` (encodage one-hot des variables catégorielles) :

| Modèle | Cible | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Retard corridor** | Retard sur le trajet routier Dakar → destination | 0,851 | 0,818 | 0,796 | 0,807 | **0,933** |
| **Dépassement franchise douane** | Séjour au port > 72h (seuil de franchise avant facturation) | 0,725 | 0,728 | 0,686 | 0,706 | **0,789** |

**Facteurs déterminants identifiés par SHAP :**
- *Retard corridor* : distance jusqu'à destination, saison des pluies, congestion du terminal, fiabilité du transporteur.
- *Dépassement franchise* : statut douanier (un dossier bloqué double le temps de séjour moyen : 122h vs 61h), congestion du terminal.

**Impact business chiffré (EDA sur le dataset simulé)** :
- Le quartile d'expéditions le plus lent coûte en moyenne **8,7 fois plus
  cher** en détention que le quartile le plus rapide.
- Le taux de retard corridor passe de **25 % en saison sèche à 65 % en
  saison des pluies**.

---

## Structure du projet

```
Projet_Corridor_Risk_Intelligence/
├── data/
│   └── corridor_risk_dataset.csv        # dataset simulé (6000 expéditions)
├── notebooks/
│   ├── 01_generation_dataset.py         # génération du dataset simulé
│   └── 02_modelisation.py               # entraînement des 2 modèles retenus
├── app.py                               # application Streamlit (Command Center)
├── modele_retard_corridor.pkl
├── modele_depassement_franchise.pkl
├── feature_meta.pkl
├── requirements.txt
└── README.md
```

L'application permet de simuler une expédition (terminal, transporteur,
destination, statut douanier, congestion...) et affiche en retour :
les deux scores de risque, une estimation du coût de détention évitable,
l'explication SHAP des facteurs déterminants, et une recommandation d'action.



Projet réalisé par Jeannette Grace Nkawara Dasylva 