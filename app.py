"""
App Streamlit – Diagnóstico asistido de cáncer de mama (ACA 1, CUN)
Modelo: SVM (kernel RBF) entrenado sobre UCI Breast Cancer Wisconsin (Diagnostic).
Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (COLOR_BENIGNO, COLOR_MALIGNO, COLOR_OSCURO, GRUPOS, RASGOS,
                   EJEMPLO_PATH, cargar_datos, cargar_metadata, cargar_modelo,
                   grupo_de, nombre_es, predecir, rasgo_de, validar_columnas)

st.set_page_config(page_title="Diagnóstico Cáncer de Mama · ML", page_icon="🩺",
                   layout="wide")

modelo = cargar_modelo()
meta = cargar_metadata()
df = cargar_datos()
FEATURES = meta["features"]
RANGOS = meta["rangos"]
PALETA = {"Benigno": COLOR_BENIGNO, "Maligno": COLOR_MALIGNO}

# ----------------------------------------------------------------------------
# Estado de los controles de entrada (un valor por variable)
# ----------------------------------------------------------------------------
def _fijar_valores(fuente: str):
    """Callback: carga en los sliders un caso de referencia."""
    if fuente == "aleatorio":
        fila = df.sample(1).iloc[0]
        st.session_state["caso_real"] = fila["Diagnóstico"]
        valores = {c: float(fila[c]) for c in FEATURES}
    else:
        st.session_state["caso_real"] = None
        clave = {"general": "mediana", "benigno": "mediana_benigno",
                 "maligno": "mediana_maligno"}[fuente]
        valores = {c: RANGOS[c][clave] for c in FEATURES}
    for c, v in valores.items():
        st.session_state[f"in_{c}"] = float(np.clip(v, RANGOS[c]["min"], RANGOS[c]["max"]))


if "in_" + FEATURES[0] not in st.session_state:
    _fijar_valores("general")


def _paso(c: str) -> float:
    rango = RANGOS[c]["max"] - RANGOS[c]["min"]
    return float(10 ** np.floor(np.log10(rango / 200)))


def _formato(c: str) -> str:
    decimales = max(0, int(-np.log10(_paso(c))))
    return f"%.{decimales}f"


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Diagnóstico asistido")
    st.caption("Cáncer de mama · Machine Learning clásico")
    pagina = st.radio("Navegación", [
        "🏠 Inicio", "🔍 Exploración de datos", "🧪 Predicción individual",
        "📂 Predicción por lotes", "📊 Desempeño del modelo", "ℹ️ Acerca de"],
        label_visibility="collapsed")
    st.divider()
    umbral = st.slider("Umbral de decisión (maligno si prob. ≥ umbral)",
                       0.05, 0.95, 0.50, 0.05,
                       help="Bajarlo detecta más malignos (menos falsos negativos) "
                            "a cambio de más falsos positivos.")
    st.divider()
    st.caption(f"Modelo: **{meta['modelo']}**  \n"
               f"F1 prueba: **{meta['metricas_test']['F1-score']:.3f}** · "
               f"AUC: **{meta['metricas_test']['ROC-AUC']:.3f}**")
    st.caption("CUN · Especialización en IA · ACA 1")

# ============================================================================
# 🏠 INICIO
# ============================================================================
if pagina == "🏠 Inicio":
    st.title("Diagnóstico asistido de cáncer de mama con Machine Learning")
    st.markdown(
        "Esta aplicación estima si un tumor de mama es **benigno** o **maligno** a partir "
        "de 30 mediciones del núcleo celular obtenidas en una biopsia por aspiración con "
        "aguja fina (FNA). El modelo es una **Máquina de Vectores de Soporte (SVM)** "
        "entrenada con el dataset *UCI Breast Cancer Wisconsin (Diagnostic)*.")

    m = meta["metricas_test"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['Accuracy']:.1%}")
    c2.metric("Precision", f"{m['Precision']:.1%}")
    c3.metric("Recall", f"{m['Recall']:.1%}")
    c4.metric("F1-score", f"{m['F1-score']:.3f}")
    c5.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    st.caption(f"Métricas sobre el conjunto de prueba ({meta['n_test']} casos nunca vistos).")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 Objetivo")
        st.write("Clasificar tumores de mama como benignos o malignos con un modelo de ML "
                 "clásico y ponerlo a disposición del usuario final en una interfaz "
                 "interactiva.")
        st.subheader("❓ Pregunta orientadora")
        st.write("¿Cómo transformar las mediciones celulares de una biopsia en una "
                 "predicción confiable del tipo de tumor y presentarla de forma "
                 "accesible para el personal de salud?")
    with col_b:
        st.subheader("🛠 Ruta de aprendizaje")
        pasos = [("1. EDA", "569 casos, 30 variables, 37 % malignos"),
                 ("2. Limpieza", "0 nulos, 0 duplicados, outliers conservados, escalado en Pipeline"),
                 ("3. Modelado", "6 algoritmos + GridSearchCV → SVM-RBF (C = 5)"),
                 ("4. Evaluación", "Accuracy, Precision, Recall, F1, AUC, matriz de confusión"),
                 ("5. Despliegue", "Esta app en Streamlit")]
        for t, d in pasos:
            st.markdown(f"**{t}** — {d}")

    st.info("👉 Ve a **🧪 Predicción individual** en el menú lateral para probar el modelo.")
    st.warning("⚠️ Herramienta académica de apoyo. **No reemplaza** el diagnóstico "
               "histopatológico ni el criterio de un profesional de la salud.")

# ============================================================================
# 🔍 EXPLORACIÓN
# ============================================================================
elif pagina == "🔍 Exploración de datos":
    st.title("🔍 Exploración de datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos", len(df))
    c2.metric("Variables", len(FEATURES))
    c3.metric("Benignos", int((df["diagnostico"] == 0).sum()))
    c4.metric("Malignos", int((df["diagnostico"] == 1).sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Relación entre variables",
                                      "Correlaciones", "Datos"])
    opciones = {nombre_es(c): c for c in FEATURES}

    with tab1:
        col_sel, col_g = st.columns([1, 3])
        with col_sel:
            var = opciones[st.selectbox("Variable", list(opciones),
                                        index=FEATURES.index("worst concave points"))]
            tipo = st.radio("Gráfico", ["Histograma", "Caja (boxplot)"])
            st.caption(RASGOS[rasgo_de(var)][1])
        with col_g:
            if tipo == "Histograma":
                fig = px.histogram(df, x=var, color="Diagnóstico", barmode="overlay",
                                   opacity=0.65, nbins=40, color_discrete_map=PALETA)
            else:
                fig = px.box(df, x="Diagnóstico", y=var, color="Diagnóstico",
                             points="outliers", color_discrete_map=PALETA)
            fig.update_layout(height=420, title=nombre_es(var))
            st.plotly_chart(fig, width="stretch")

        conteo = df["Diagnóstico"].value_counts().reset_index()
        fig = px.pie(conteo, names="Diagnóstico", values="count", hole=0.55,
                     color="Diagnóstico", color_discrete_map=PALETA,
                     title="Balance de clases")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        x = opciones[c1.selectbox("Eje X", list(opciones), index=FEATURES.index("mean radius"))]
        y = opciones[c2.selectbox("Eje Y", list(opciones), index=FEATURES.index("mean texture"))]
        fig = px.scatter(df, x=x, y=y, color="Diagnóstico", opacity=0.7,
                         color_discrete_map=PALETA,
                         labels={x: nombre_es(x), y: nombre_es(y)})
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        grupo = st.radio("Grupo de variables", list(GRUPOS.values()), horizontal=True)
        clave = [k for k, v in GRUPOS.items() if v == grupo][0]
        cols = [c for c in FEATURES if grupo_de(c) == clave] + ["diagnostico"]
        corr = df[cols].corr().round(2)
        etiquetas = [RASGOS[rasgo_de(c)][0] if c != "diagnostico" else "DIAGNÓSTICO"
                     for c in cols]
        fig = px.imshow(corr.values, x=etiquetas, y=etiquetas, text_auto=True,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=560)
        st.plotly_chart(fig, width="stretch")
        st.caption("Radio, perímetro y área están casi perfectamente correlacionados "
                   "(multicolinealidad). La SVM con regularización maneja esto sin "
                   "eliminar variables.")

    with tab4:
        st.dataframe(df.drop(columns="diagnostico"), width="stretch", height=420)
        st.download_button("⬇️ Descargar CSV", df.drop(columns="Diagnóstico").to_csv(index=False),
                           "breast_cancer_wisconsin.csv", "text/csv")

# ============================================================================
# 🧪 PREDICCIÓN INDIVIDUAL
# ============================================================================
elif pagina == "🧪 Predicción individual":
    st.title("🧪 Predicción individual")
    st.write("Ajusta las mediciones del núcleo celular o carga un caso de referencia. "
             "La predicción se actualiza automáticamente.")

    b1, b2, b3, b4 = st.columns(4)
    b1.button("Caso típico benigno", on_click=_fijar_valores, args=("benigno",),
              width="stretch")
    b2.button("Caso típico maligno", on_click=_fijar_valores, args=("maligno",),
              width="stretch")
    b3.button("🎲 Paciente real aleatorio", on_click=_fijar_valores, args=("aleatorio",),
              width="stretch")
    b4.button("↺ Restablecer (medianas)", on_click=_fijar_valores, args=("general",),
              width="stretch")

    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        tabs = st.tabs([f"{v}" for v in GRUPOS.values()])
        for tab, clave in zip(tabs, GRUPOS):
            with tab:
                feats = [c for c in FEATURES if grupo_de(c) == clave]
                cols = st.columns(2)
                for i, c in enumerate(feats):
                    r = RANGOS[c]
                    cols[i % 2].slider(RASGOS[rasgo_de(c)][0], min_value=float(r["min"]),
                                       max_value=float(r["max"]), step=_paso(c),
                                       format=_formato(c), key=f"in_{c}",
                                       help=RASGOS[rasgo_de(c)][1])

    X_in = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in FEATURES}])
    res = predecir(modelo, X_in, umbral).iloc[0]
    proba = float(res["Probabilidad maligno"])
    es_maligno = res["Predicción"] == "Maligno"

    with col_out:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number={"suffix": " %", "font": {"size": 44}},
            title={"text": "Probabilidad de malignidad"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": COLOR_MALIGNO if es_maligno else COLOR_BENIGNO},
                   "steps": [{"range": [0, umbral * 100], "color": "#e8f5f3"},
                             {"range": [umbral * 100, 100], "color": "#fdece7"}],
                   "threshold": {"line": {"color": COLOR_OSCURO, "width": 4},
                                 "thickness": 0.8, "value": umbral * 100}}))
        fig.update_layout(height=290, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, width="stretch")

        if es_maligno:
            st.error(f"### 🔴 Predicción: MALIGNO\nProbabilidad {proba:.1%} ≥ umbral {umbral:.0%}")
        else:
            st.success(f"### 🟢 Predicción: BENIGNO\nProbabilidad {proba:.1%} < umbral {umbral:.0%}")

        confianza = abs(proba - umbral)
        if confianza < 0.15:
            st.warning("Caso **cercano al umbral**: la predicción es incierta. "
                       "Se recomienda revisión adicional.")
        if st.session_state.get("caso_real"):
            ok = st.session_state["caso_real"] == res["Predicción"]
            st.info(f"Diagnóstico real de este paciente: **{st.session_state['caso_real']}** "
                    f"{'✅ coincide' if ok else '❌ no coincide'}")

    st.subheader("¿Cómo se compara este caso?")
    top = sorted(meta["importancias"], key=meta["importancias"].get, reverse=True)[:8]
    comp = pd.DataFrame({
        "Variable": [nombre_es(c) for c in top],
        "Este caso": [st.session_state[f"in_{c}"] / RANGOS[c]["max"] for c in top],
        "Mediana benigno": [RANGOS[c]["mediana_benigno"] / RANGOS[c]["max"] for c in top],
        "Mediana maligno": [RANGOS[c]["mediana_maligno"] / RANGOS[c]["max"] for c in top],
    }).melt(id_vars="Variable", var_name="Serie", value_name="Valor relativo")
    fig = px.bar(comp, y="Variable", x="Valor relativo", color="Serie", barmode="group",
                 orientation="h", color_discrete_map={"Este caso": COLOR_OSCURO,
                 "Mediana benigno": COLOR_BENIGNO, "Mediana maligno": COLOR_MALIGNO})
    fig.update_layout(height=430, xaxis_title="Valor / máximo observado",
                      yaxis={"autorange": "reversed"}, legend_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Las 8 variables más importantes según la importancia por permutación. "
               "Si las barras oscuras se parecen más a las rojas, el caso se asemeja al "
               "perfil maligno.")

# ============================================================================
# 📂 PREDICCIÓN POR LOTES
# ============================================================================
elif pagina == "📂 Predicción por lotes":
    st.title("📂 Predicción por lotes")
    st.write("Sube un archivo **CSV** con las 30 columnas del dataset (mismos nombres en "
             "inglés) y obtén la predicción para cada paciente.")
    with open(EJEMPLO_PATH, "rb") as f:
        st.download_button("⬇️ Descargar archivo de ejemplo (10 pacientes)", f,
                           "ejemplo_pacientes.csv", "text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:  # archivo corrupto o con otro separador
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()
        faltan = validar_columnas(datos, FEATURES)
        if faltan:
            st.error(f"Faltan {len(faltan)} columnas requeridas: {', '.join(faltan[:6])}"
                     f"{'…' if len(faltan) > 6 else ''}")
            st.stop()
        X_lote = datos[FEATURES].apply(pd.to_numeric, errors="coerce")
        filas_invalidas = X_lote.isnull().any(axis=1)
        if filas_invalidas.any():
            st.warning(f"{int(filas_invalidas.sum())} fila(s) con valores vacíos o no "
                       "numéricos fueron omitidas.")
            X_lote = X_lote[~filas_invalidas]
        if X_lote.empty:
            st.error("No quedan filas válidas para predecir."); st.stop()

        salida = pd.concat([predecir(modelo, X_lote, umbral), X_lote], axis=1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pacientes evaluados", len(salida))
        c2.metric("Predichos malignos", int((salida["Predicción"] == "Maligno").sum()))
        c3.metric("Predichos benignos", int((salida["Predicción"] == "Benigno").sum()))

        st.dataframe(salida.style.map(
            lambda v: f"color: {COLOR_MALIGNO}; font-weight: bold" if v == "Maligno"
            else (f"color: {COLOR_BENIGNO}" if v == "Benigno" else ""),
            subset=["Predicción"]), width="stretch")
        st.download_button("⬇️ Descargar resultados", salida.to_csv(index=False),
                           "predicciones.csv", "text/csv", type="primary")

# ============================================================================
# 📊 DESEMPEÑO
# ============================================================================
elif pagina == "📊 Desempeño del modelo":
    st.title("📊 Desempeño del modelo")
    st.write(f"**Modelo final:** {meta['modelo']} · **Hiperparámetros:** "
             + ", ".join(f"`{k} = {v}`" for k, v in meta["parametros"].items()))
    m = meta["metricas_test"]
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    t1, t2, t3, t4 = st.tabs(["Matriz de confusión y ROC", "Comparación de modelos",
                              "Efecto del umbral", "Importancia de variables"])
    with t1:
        c1, c2 = st.columns(2)
        cm = meta["matriz_confusion"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = px.imshow(z, x=["Pred. Benigno", "Pred. Maligno"],
                        y=["Real Benigno", "Real Maligno"], text_auto=True,
                        color_continuous_scale="Blues")
        fig.update_traces(textfont_size=22)
        fig.update_layout(title="Matriz de confusión (umbral 0.5)", height=400,
                          coloraxis_showscale=False)
        c1.plotly_chart(fig, width="stretch")
        c1.caption(f"{cm['fn']} falsos negativos y {cm['fp']} falsos positivos en "
                   f"{meta['n_test']} casos de prueba.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=meta["roc"]["fpr"], y=meta["roc"]["tpr"], fill="tozeroy",
                                 line=dict(color=COLOR_MALIGNO, width=3),
                                 name=f"SVM (AUC = {m['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                                 name="Aleatorio"))
        fig.update_layout(title="Curva ROC", height=400, xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos (Recall)",
                          legend=dict(x=0.45, y=0.1))
        c2.plotly_chart(fig, width="stretch")

    with t2:
        cv = pd.DataFrame(meta["metricas_cv"]).T
        st.dataframe(cv.style.format("{:.4f}").highlight_max(axis=0, color="#d8f3dc"),
                     width="stretch")
        fig = px.bar(cv.reset_index().melt(id_vars="index", value_vars=["recall", "f1", "roc_auc"]),
                     x="index", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[COLOR_MALIGNO, COLOR_BENIGNO, "#e9c46a"])
        fig.update_layout(yaxis_range=[0.85, 1], xaxis_title="", yaxis_title="Puntaje",
                          legend_title="", height=400)
        st.plotly_chart(fig, width="stretch")
        st.caption("Validación cruzada estratificada de 5 pliegues con hiperparámetros por "
                   "defecto. Luego Regresión Logística y SVM se ajustaron con GridSearchCV "
                   "y ganó SVM (F1 CV = 0.973).")

    with t3:
        st.dataframe(pd.DataFrame(meta["umbral"]).set_index("Umbral"),
                     width="stretch")
        st.write("Bajar el umbral reduce los **falsos negativos** (malignos no detectados), "
                 "que son el error más grave en tamizaje, a costa de algunos falsos "
                 "positivos. Usa el control del menú lateral para aplicarlo en las "
                 "predicciones.")

    with t4:
        imp = (pd.Series(meta["importancias"]).sort_values(ascending=False).head(12)
               .rename(index=nombre_es).sort_values())
        fig = px.bar(imp, orientation="h", color_discrete_sequence=[COLOR_OSCURO])
        fig.update_layout(height=460, showlegend=False, xaxis_title="Caída media del F1",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Importancia por permutación: cuánto empeora el F1 si se desordena "
                   "aleatoriamente cada variable.")

# ============================================================================
# ℹ️ ACERCA DE
# ============================================================================
else:
    st.title("ℹ️ Acerca del proyecto")
    st.markdown("""
**Corporación Unificada Nacional de Educación Superior (CUN)** · Escuela de Ingeniería
Especialización en Inteligencia Artificial · Materia: APRENDIZAJE AUTOMÁTICO/54ES1/SEGUNDO BLOQUE/26ES4
Formador: Hamilton Rivera · Actividad: ACA N° 1 · Fecha: 1 de noviembre de 2026  
Integrantes: Duvan Daniel Gómez Alían, Jeisson Jesith Ortega Lozano, Dubier Alberto Quintero Gómez y Duberney Arboleda Luján

**Pilar:** Salud – detección y diagnóstico asistido de enfermedades.

**Stack:** Python · pandas · scikit-learn · Streamlit · Plotly · joblib

#### Referencias (APA 7)
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. En *Biomedical Image Processing and Biomedical Visualization* (Vol. 1905, pp. 861–870). SPIE. https://doi.org/10.1117/12.148698
- Wolberg, W., Street, N., & Mangasarian, O. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
""")
    st.warning("Esta herramienta tiene fines académicos. No debe usarse para tomar "
               "decisiones clínicas reales.")
