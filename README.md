# Diagnóstico asistido de cáncer de mama con Machine Learning clásico

**ACA N° 1 – Aprendizaje Automático** · CUN · Especialización en Inteligencia Artificial
Materia: APRENDIZAJE AUTOMÁTICO/54ES1/SEGUNDO BLOQUE/26ES4 · Formador: Hamilton Rivera · Fecha: 1 de noviembre de 2026  
Integrantes: Duvan Daniel Gómez Alían, Jeisson Jesith Ortega Lozano, Dubier Alberto Quintero Gómez y Duberney Arboleda Luján  
Pilar: **Salud** · Dataset: **UCI Breast Cancer Wisconsin (Diagnostic)**

Modelo final: **SVM con kernel RBF** (C = 5, gamma = "scale") dentro de un `Pipeline` con `StandardScaler`.

| Métrica (prueba, 114 casos) | Valor |
|---|---|
| Accuracy | 0.9737 |
| Precision | 1.0000 |
| Recall | 0.9286 |
| F1-score | 0.9630 |
| ROC-AUC | 0.9947 |
| Especificidad | 1.0000 |

## Estructura del proyecto

```
proyecto/
├── ACA1_BreastCancer_ML.ipynb   ← Entregable 1: notebook (EDA → limpieza → modelado → métricas)
├── app.py                       ← Entregable 3: interfaz Streamlit
├── utils.py                     ← funciones auxiliares de la app
├── requirements.txt             ← dependencias (versiones fijas)
├── .streamlit/config.toml       ← tema visual de la app
├── models/
│   ├── modelo_cancer_mama.joblib  ← pipeline entrenado (escalador + SVM)
│   └── metadata.json              ← métricas, rangos y curvas para la app
├── data/
│   ├── breast_cancer_wisconsin.csv
│   └── ejemplo_pacientes.csv      ← archivo para probar la predicción por lotes
├── figures/                     ← gráficos generados por el notebook y capturas de la app
└── GUION_VIDEO.md               ← Entregable 4: guion de la sustentación
```

## 1. Ejecutar en local

Requisitos: Python 3.11 o superior.

```bash
# Crear y activar entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install jupyter seaborn matplotlib   # solo para el notebook

# (Opcional) volver a entrenar: abre y ejecuta todo el notebook
jupyter notebook ACA1_BreastCancer_ML.ipynb

# Lanzar la app
streamlit run app.py
```

La app abre en `http://localhost:8501`.

> **Importante:** el modelo `.joblib` se guardó con scikit-learn 1.9.1. Si instalas otra versión y aparece un error al cargarlo, ejecuta de nuevo el notebook para regenerar el archivo con tu versión.

## 2. Publicar gratis en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube **todo** el contenido de esta carpeta (incluidas `models/` y `data/`).
2. Entra a https://share.streamlit.io e inicia sesión con tu cuenta de GitHub.
3. Clic en **Create app** → elige el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings** selecciona Python 3.12 o superior.
5. Clic en **Deploy**. En 2–4 minutos tendrás un enlace público tipo `https://<tu-app>.streamlit.app`.
6. Copia ese enlace al póster y al video.

## 3. Páginas de la app

- **Inicio:** objetivo, pregunta orientadora, ruta metodológica y métricas.
- **Exploración de datos:** histogramas, boxplots, dispersión y correlaciones interactivas.
- **Predicción individual:** 30 controles deslizantes (agrupados en Media / Error estándar / Peor valor), casos de ejemplo y medidor de probabilidad.
- **Predicción por lotes:** sube un CSV y descarga los resultados.
- **Desempeño del modelo:** matriz de confusión, curva ROC, comparación de modelos, efecto del umbral e importancia de variables.
- **Umbral de decisión** (barra lateral): bajarlo detecta más malignos a cambio de más falsos positivos.

## Referencias (APA 7)

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. En *Biomedical Image Processing and Biomedical Visualization* (Vol. 1905, pp. 861–870). SPIE. https://doi.org/10.1117/12.148698
- Wolberg, W., Street, N., & Mangasarian, O. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

> ⚠️ Proyecto académico. El modelo es una herramienta de apoyo y no reemplaza el diagnóstico médico.
