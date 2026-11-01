"""Funciones auxiliares de la app: carga del modelo, datos y traducciones."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_cancer_mama.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"
DATA_PATH = BASE_DIR / "data" / "breast_cancer_wisconsin.csv"
EJEMPLO_PATH = BASE_DIR / "data" / "ejemplo_pacientes.csv"

COLOR_BENIGNO = "#2a9d8f"
COLOR_MALIGNO = "#e76f51"
COLOR_OSCURO = "#264653"

# Traducción de los 10 rasgos base del núcleo celular
RASGOS = {
    "radius": ("Radio", "Distancia media del centro al borde del núcleo"),
    "texture": ("Textura", "Desviación estándar de la escala de grises"),
    "perimeter": ("Perímetro", "Longitud del contorno del núcleo"),
    "area": ("Área", "Superficie del núcleo (píxeles²)"),
    "smoothness": ("Suavidad", "Variación local en la longitud del radio"),
    "compactness": ("Compacidad", "Perímetro² / área − 1"),
    "concavity": ("Concavidad", "Severidad de las partes cóncavas del contorno"),
    "concave points": ("Puntos cóncavos", "Número de partes cóncavas del contorno"),
    "symmetry": ("Simetría", "Simetría del núcleo"),
    "fractal dimension": ("Dimensión fractal", "Aproximación de la 'línea de costa' − 1"),
}

GRUPOS = {
    "mean": "Media",
    "error": "Error estándar",
    "worst": "Peor valor",
}


def grupo_de(feature: str) -> str:
    if feature.startswith("mean "):
        return "mean"
    if feature.startswith("worst "):
        return "worst"
    return "error"


def rasgo_de(feature: str) -> str:
    return feature.replace("mean ", "").replace("worst ", "").replace(" error", "")


def nombre_es(feature: str) -> str:
    """'worst concave points' -> 'Puntos cóncavos (peor valor)'."""
    rasgo = RASGOS[rasgo_de(feature)][0]
    return f"{rasgo} ({GRUPOS[grupo_de(feature)].lower()})"


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_metadata() -> dict:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Diagnóstico"] = df["diagnostico"].map({0: "Benigno", 1: "Maligno"})
    return df


def predecir(modelo, X: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    """Devuelve probabilidad de malignidad y clase según el umbral elegido."""
    proba = modelo.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "Probabilidad maligno": proba.round(4),
        "Predicción": ["Maligno" if p >= umbral else "Benigno" for p in proba],
    }, index=X.index)


def validar_columnas(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Lista de columnas requeridas que faltan en el archivo cargado."""
    return [c for c in features if c not in df.columns]
