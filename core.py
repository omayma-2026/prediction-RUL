"""
core.py – Logique métier du projet RUL (NASA C-MAPSS FD001)

Ce module regroupe tout ce qui n'est pas de l'interface :
chargement des données, construction de la RUL, variables glissantes,
entraînement / comparaison des modèles, métriques, sauvegarde et export Excel.

Utilisation en ligne de commande (entraîne et sauvegarde le modèle) :
    python core.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "rul_model.joblib"
META_PATH = MODEL_DIR / "meta.json"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
CAP = 125                 # plafonnement de la RUL (dégradation linéaire par morceaux)
WINDOWS = (5, 10)         # fenêtres glissantes (en cycles)
CRIT_DEFAULT = 30         # seuil critique (cycles)
WARN_DEFAULT = 60         # seuil d'attention (cycles)


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
RAW_URL = "https://raw.githubusercontent.com/omayma-2026/prediction-RUL/main/"


def _locate(name: str, data_dir: Path = DATA_DIR) -> str:
    """Cherche le fichier localement ; sinon le télécharge depuis GitHub."""
    for base in (Path(data_dir), BASE_DIR, BASE_DIR / "data", Path.cwd()):
        if (base / name).exists():
            return str(base / name)
    hits = list(BASE_DIR.rglob(name))
    if hits:
        return str(hits[0])
    return RAW_URL + name


def load_raw(data_dir: Path = DATA_DIR):
    """Charge train / test / RUL réelle du test (FD001)."""
    kw = dict(sep=r"\s+", header=None)
    train = pd.read_csv(_locate("train_FD001.txt", data_dir), names=COLS, **kw)
    test = pd.read_csv(_locate("test_FD001.txt", data_dir), names=COLS, **kw)
    rul = pd.read_csv(_locate("RUL_FD001.txt", data_dir), names=["RUL"], **kw)
    return train, test, rul


def add_rul(train: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    """RUL = dernier cycle du moteur - cycle courant, puis plafonnée."""
    train = train.copy()
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    train["RUL_clip"] = train["RUL"].clip(upper=cap)
    return train


def select_sensors(train: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Garde les colonnes capteurs/réglages dont l'écart-type n'est pas quasi nul."""
    candidates = [c for c in COLS if c not in ("unit", "cycle")]
    stds = train[candidates].std()
    return list(stds[stds >= threshold].index)


# --------------------------------------------------------------------------- #
# Variables sur fenêtres glissantes
# --------------------------------------------------------------------------- #
def get_feature_cols(sensors, windows=WINDOWS) -> list[str]:
    feats = ["cycle"] + list(sensors)
    for w in windows:
        feats += [f"{c}_mean{w}" for c in sensors]
        feats += [f"{c}_std{w}" for c in sensors]
    return feats


def add_features(df: pd.DataFrame, sensors, windows=WINDOWS) -> pd.DataFrame:
    """Moyenne et écart-type glissants par moteur (uniquement sur le passé)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[list(sensors)]
    parts = [df]
    for w in windows:
        mean = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        std = g.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        mean.columns = [f"{c}_mean{w}" for c in sensors]
        std.columns = [f"{c}_std{w}" for c in sensors]
        parts += [mean, std]
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def nasa_score(y_true, y_pred) -> float:
    """Score asymétrique NASA : pénalise davantage les prédictions trop optimistes."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA score": nasa_score(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    return {
        "Régression linéaire": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
        ),
    }


def run_training(data_dir: Path = DATA_DIR) -> dict:
    """Entraîne les 3 modèles, les évalue sur le test et renvoie tout le nécessaire."""
    train, test, rul = load_raw(data_dir)
    train = add_rul(train)
    sensors = select_sensors(train)
    features = get_feature_cols(sensors, WINDOWS)

    train_f = add_features(train, sensors, WINDOWS)
    train_f["RUL_clip"] = (
        train_f.groupby("unit")["cycle"].transform("max") - train_f["cycle"]
    ).clip(upper=CAP)
    X_train, y_train = train_f[features], train_f["RUL_clip"]

    test_f = add_features(test, sensors, WINDOWS)
    last = test_f.groupby("unit").tail(1)
    X_test = last[features]
    y_test = rul["RUL"].clip(upper=CAP).values

    models, results = build_models(), {}
    preds = pd.DataFrame(
        {"moteur": last["unit"].values, "cycles_observes": last["cycle"].values, "RUL_reelle": y_test}
    )
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = np.clip(model.predict(X_test), 0, CAP)
        preds[f"RUL_predite_{name}"] = p.round(1)
        results[name] = evaluate(y_test, p)

    results_df = pd.DataFrame(results).T
    best = results_df["RMSE"].idxmin()

    rf = models["Random Forest"]
    importance = (
        pd.Series(rf.feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(30)
        .rename("importance")
        .rename_axis("variable")
        .reset_index()
    )
    meta = {
        "sensors": sensors,
        "windows": list(WINDOWS),
        "features": features,
        "cap": CAP,
        "model": best,
        "results": results,
        "importance": importance.values.tolist(),
        "preds": preds.to_dict(orient="records"),
    }
    return {"models": models, "best": best, "meta": meta}


def save_artifacts(out: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(out["models"][out["best"]], MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps(out["meta"], ensure_ascii=False))


def train_and_save(data_dir: Path = DATA_DIR) -> dict:
    out = run_training(data_dir)
    save_artifacts(out)
    return out["meta"]


def load_artifacts():
    """Charge modèle + métadonnées ; entraîne automatiquement s'ils sont absents."""
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())
        except Exception:
            pass  # modèle illisible (version de scikit-learn différente) : on réentraîne
    train_and_save()
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


# --------------------------------------------------------------------------- #
# Prédiction / suivi
# --------------------------------------------------------------------------- #
def load_test_features(meta: dict, data_dir: Path = DATA_DIR):
    """Variables du jeu de test + RUL finale réelle de chaque moteur."""
    _, test, rul = load_raw(data_dir)
    feat = add_features(test, meta["sensors"], tuple(meta["windows"]))
    return feat, rul["RUL"].values


def predict_rul(model, meta: dict, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(df[meta["features"]]), 0, meta["cap"])


def predict_fleet(model, meta: dict, feat: pd.DataFrame) -> pd.DataFrame:
    last = feat.groupby("unit").tail(1)
    return pd.DataFrame(
        {
            "moteur": last["unit"].values,
            "cycles_observes": last["cycle"].values,
            "RUL_predite": predict_rul(model, meta, last).round(1),
        }
    )


def status(rul: float, crit: float = CRIT_DEFAULT, warn: float = WARN_DEFAULT):
    """Renvoie (libellé, couleur) selon les seuils."""
    if rul < crit:
        return "Critique", "#d32f2f"
    if rul < warn:
        return "Attention", "#f57c00"
    return "Normal", "#2e7d32"


# --------------------------------------------------------------------------- #
# Export Excel
# --------------------------------------------------------------------------- #
def _style_workbook(wb, highlight_row_label: str | None = None) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="1F3A5F")
    best_fill = PatternFill("solid", fgColor="DCEFD9")
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="Arial", size=10)
            if highlight_row_label and ws.title == "Métriques" and row[0].value == highlight_row_label:
                for cell in row:
                    cell.fill = best_fill
                    cell.font = Font(name="Arial", size=10, bold=True)
        for i, col in enumerate(ws.columns, start=1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(width + 3, 12), 45)
        ws.freeze_panes = "A2"


def build_excel(meta: dict) -> bytes:
    """Rapport Excel : métriques, prédictions par moteur, importance des variables."""
    results = pd.DataFrame(meta["results"]).T.round(2)
    preds = pd.DataFrame(meta["preds"])
    best = meta["model"]
    preds[f"Erreur_{best}"] = (preds[f"RUL_predite_{best}"] - preds["RUL_reelle"]).round(1)
    preds["Statut"] = [status(v)[0] for v in preds[f"RUL_predite_{best}"]]
    importance = pd.DataFrame(meta["importance"], columns=["variable", "importance"]).round(4)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        results.to_excel(xw, sheet_name="Métriques", index_label="Modèle")
        preds.to_excel(xw, sheet_name="Prédictions", index=False)
        importance.to_excel(xw, sheet_name="Importance variables", index=False)
        _style_workbook(xw.book, highlight_row_label=best)
    return buf.getvalue()


if __name__ == "__main__":
    m = train_and_save()
    print("Modèle sauvegardé :", m["model"])
    print(pd.DataFrame(m["results"]).T.round(2))
