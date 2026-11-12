# ✈️ RUL Monitor – Prédiction de la durée de vie restante de moteurs turbofan

Projet de maintenance prédictive sur le jeu de données **NASA C-MAPSS (FD001)** :
prédire le nombre de cycles restants avant la panne (RUL, *Remaining Useful Life*) et suivre l'état d'une flotte de moteurs dans une application web.

## Fonctionnalités
- Construction de la RUL (plafonnée à 125 cycles) et sélection des capteurs informatifs (14 sur 24)
- Variables sur **fenêtres glissantes** (moyenne et écart-type sur 5 et 10 cycles) → 71 variables
- Comparaison de 3 modèles : **régression linéaire, Random Forest, Gradient Boosting**
- Application **Streamlit** : suivi d'un moteur cycle par cycle, vue flotte, comparaison des modèles, export Excel

## Résultats (test FD001, 100 moteurs)
| Modèle | RMSE | MAE | R² | Score NASA |
|---|---|---|---|---|
| Régression linéaire | 20.62 | 15.67 | 0.74 | 986 |
| Random Forest | 17.85 | 12.94 | 0.80 | 623 |
| **Gradient Boosting** | **17.37** | **12.34** | 0.81 | 675 |

Évaluation au **dernier cycle observé** de chaque moteur de test, comparé à la RUL réelle de `RUL_FD001.txt`.

## Structure
```
├── app.py            # interface Streamlit
├── core.py           # données, variables, entraînement, métriques, export Excel
├── requirements.txt
├── data/             # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
├── models/           # rul_model.joblib + meta.json (générés par core.py)
└── .streamlit/config.toml
```

## Lancer en local
```bash
pip install -r requirements.txt
python core.py            # (optionnel) réentraîne et sauvegarde le modèle
streamlit run app.py
```
Si `models/` est vide, l'application entraîne le modèle automatiquement au premier lancement (environ 1 minute).

## Déployer (Streamlit Community Cloud)
1. Pousser ce dossier sur GitHub.
2. Sur https://share.streamlit.io : *New app* → choisir le dépôt, branche `main`, fichier `app.py`.

## Source des données
NASA Prognostics Center of Excellence – *Turbofan Engine Degradation Simulation (C-MAPSS)*.
Référence : A. Saxena, K. Goebel, D. Simon, N. Eklund, « Damage propagation modeling for aircraft engine run-to-failure simulation », PHM 2008.
