import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore

# Configuración visual
sns.set_theme(style="whitegrid", context="talk", palette="viridis") # Cambiado a paleta viridis para mejor contraste

# Rutas
archivo = "modulo4/entrada/voter_turnout_socioeconomic_csv.csv"
carpeta_base = "modulo4/salida/reto20_mejorado" # Nueva carpeta para las salidas mejoradas
os.makedirs(carpeta_base, exist_ok=True)

# ========================
# 🔹 CARGA Y LIMPIEZA DE DATOS
# ========================
try:
    df = pd.read_csv(archivo)
except FileNotFoundError:
    print(f"Error: El archivo no se encuentra en la ruta especificada: {archivo}")
    print("Por favor, asegúrate de que el archivo 'voter_turnout_socioeconomic_csv.csv' esté en la carpeta 'modulo4/entrada'.")
    exit()

df = df.drop_duplicates()
df = df.dropna()

# Convertir columnas categóricas
for col in ["region", "education_level"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Eliminar outliers numéricos usando z-score (Aplicado solo a columnas numéricas relevantes)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if not numeric_cols.empty:
    z_scores = np.abs(zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
else:
    print("Advertencia: No se encontraron columnas numéricas para aplicar la eliminación de outliers por z-score.")

# Validar porcentajes en rango [0, 100]
df = df[(df["voter_turnout_pct"].between(0, 100)) & (df["trust_government_pct"].between(0, 100))]

# ========================
# 🔹 FUNCIONES DE SALIDA GERENCIAL (Mismas que en el código original)
# ========================
def resumen_gerencial(titulo, objetivo, conclusion, ruta_grafico):
    print("\n" + "="*80)
    print(titulo)
    print(f"\n📌 Objetivo: {objetivo}")
    print(f"\n📊 Gráfico guardado en: {ruta_grafico}")
    print(f"\n🧠 Conclusión:\n{conclusion}")
    print("="*80)

def guardar_slide(fig, objetivo, conclusion, nombre_salida):
    # Asegurarse de que el texto no se superponga si hay muchos gráficos
    plt.figtext(0.5, -0.08, f"📌 Objetivo: {objetivo}", wrap=True, ha='center', fontsize=10, color='gray')
    plt.figtext(0.5, -0.15, f"🧠 Conclusión: {conclusion}", wrap=True, ha='center', fontsize=11)
    fig.savefig(nombre_salida, bbox_inches='tight')
    plt.close(fig) # Cerrar la figura para liberar memoria
    # No mostrar plt.show() aquí, se llama manualmente si se desea ver el gráfico al instante

# ====================================================================
# 📈 GRÁFICOS MEJORADOS Y NUEVOS (Basados en tu lista de "faltantes")
# ====================================================================

# ---
## Gráfico 1: Boxplot - Distribución y outliers de participación por nivel educativo
# ---
titulo = "🔎 ANÁLISIS 1: Distribución de Participación Electoral por Nivel Educativo"
objetivo = "Visualizar la distribución, mediana y posibles outliers de la participación electoral para cada nivel educativo."
conclusion = "Los boxplots muestran las diferencias en la dispersión y la mediana de la participación entre los grupos educativos, revelando dónde la participación es más consistente o variable."
grafico = f"{carpeta_base}/boxplot_participacion_educacion.png"
slide = f"{carpeta_base}/slide_boxplot_participacion_educacion.png"

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="education_level", y="voter_turnout_pct", ax=ax, palette="coolwarm")
ax.set_title("Distribución de la Participación Electoral por Nivel Educativo", fontsize=16)
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Participación Electoral (%)")
plt.xticks(rotation=15)
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 2: Swarmplot - Densidad de puntos de confianza por región
# ---
titulo = "🔎 ANÁLISIS 2: Confianza en el Gobierno por Región (Puntos Individuales)"
objetivo = "Mostrar la densidad de puntos individuales de confianza en el gobierno para cada región, evitando la superposición."
conclusion = "El swarmplot permite ver la concentración real de las respuestas sobre confianza dentro de cada región, identificando posibles conglomerados de datos."
grafico = f"{carpeta_base}/swarmplot_confianza_region.png"
slide = f"{carpeta_base}/slide_swarmplot_confianza_region.png"

fig, ax = plt.subplots(figsize=(12, 7))
sns.swarmplot(data=df, x="region", y="trust_government_pct", ax=ax, size=3, palette="mako")
ax.set_title("Distribución de Confianza en el Gobierno por Región", fontsize=16)
ax.set_xlabel("Región")
ax.set_ylabel("Confianza en el Gobierno (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 3: Stripplot - Dispersión simple de edad por nivel educativo
# ---
titulo = "🔎 ANÁLISIS 3: Dispersión de Edades por Nivel Educativo"
objetivo = "Visualizar la dispersión individual de las edades dentro de cada nivel educativo para detectar patrones o rangos."
conclusion = "El stripplot ofrece una vista clara de cada punto de edad para cada categoría educativa, útil para ver si hay rangos de edad predominantes en ciertos niveles educativos."
grafico = f"{carpeta_base}/stripplot_edad_educacion.png"
slide = f"{carpeta_base}/slide_stripplot_edad_educacion.png"

fig, ax = plt.subplots(figsize=(10, 6))
sns.stripplot(data=df, x="education_level", y="age", ax=ax, jitter=0.2, alpha=0.6, palette="Spectral")
ax.set_title("Dispersión de Edades por Nivel Educativo", fontsize=16)
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Edad")
plt.xticks(rotation=15)
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 4: Countplot - Conteo por categoría de nivel educativo
# ---
titulo = "🔎 ANÁLISIS 4: Conteo de Observaciones por Nivel Educativo"
objetivo = "Mostrar la cantidad de entradas (conteo) para cada categoría de nivel educativo en el dataset."
conclusion = "Este gráfico proporciona una vista rápida de la distribución del tamaño de la muestra entre los diferentes niveles educativos, indicando qué categorías tienen más datos."
grafico = f"{carpeta_base}/countplot_educacion.png"
slide = f"{carpeta_base}/slide_countplot_educacion.png"

fig, ax = plt.subplots(figsize=(9, 6))
sns.countplot(data=df, y="education_level", order=df["education_level"].value_counts().index, ax=ax, palette="plasma")
ax.set_title("Número de Observaciones por Nivel Educativo", fontsize=16)
ax.set_xlabel("Conteo")
ax.set_ylabel("Nivel Educativo")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 5: Heatmap - Correlación entre variables numéricas
# ---
titulo = "🔎 ANÁLISIS 5: Correlación entre Variables Numéricas Clave"
objetivo = "Visualizar la matriz de correlación entre las principales variables numéricas para identificar relaciones lineales."
conclusion = "El heatmap revela la fuerza y dirección de las correlaciones. Por ejemplo, si hay una fuerte correlación positiva entre 'income_usd' y 'voter_turnout_pct'."
grafico = f"{carpeta_base}/heatmap_correlacion.png"
slide = f"{carpeta_base}/slide_heatmap_correlacion.png"

# Seleccionar solo las columnas numéricas de interés para la correlación
numeric_corr_cols = ["age", "income_usd", "voter_turnout_pct", "trust_government_pct"]
corr_matrix = df[numeric_corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
ax.set_title("Matriz de Correlación de Variables Numéricas", fontsize=16)
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 6: Catplot - Múltiples tipos de gráfica categórica con columnas/facetas
# (Usando 'box' para mostrar la distribución por educación y región)
# ---
titulo = "🔎 ANÁLISIS 6: Participación Electoral por Educación y Región (Catplot)"
objetivo = "Analizar la distribución de la participación electoral en función de la educación, segmentada por región utilizando un catplot."
conclusion = "El catplot permite comparar fácilmente las distribuciones de participación para diferentes combinaciones de nivel educativo y región, mostrando cómo estos factores interactúan."
grafico = f"{carpeta_base}/catplot_participacion_educacion_region.png"
slide = f"{carpeta_base}/slide_catplot_participacion_educacion_region.png"

# Usamos kind='box' para ver la distribución y outliers por grupo
g = sns.catplot(data=df, x="education_level", y="voter_turnout_pct", col="region",
                kind="box", col_wrap=3, height=4, aspect=1.2, palette="GnBu", sharey=True)
g.set_axis_labels("Nivel Educativo", "Participación Electoral (%)")
g.set_titles("Región: {col_name}")
g.fig.suptitle("Distribución de Participación Electoral por Educación y Región", y=1.02, fontsize=16)
g.set_xticklabels(rotation=30, ha="right")
plt.tight_layout()
g.fig.savefig(grafico)
guardar_slide(g.fig, objetivo, conclusion, slide) # Pasar g.fig ya que catplot devuelve una FacetGrid
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 7: Lmplot - Regresión lineal por grupos (Participación vs Ingreso por Educación)
# ---
titulo = "🔎 ANÁLISIS 7: Relación Lineal entre Ingreso y Participación por Nivel Educativo"
objetivo = "Investigar la relación lineal entre el ingreso anual y la participación electoral, diferenciando por nivel educativo."
conclusion = "El lmplot permite visualizar si existe una tendencia lineal y si esta tendencia varía significativamente entre los diferentes niveles educativos, con su banda de confianza."
grafico = f"{carpeta_base}/lmplot_ingreso_participacion_educacion.png"
slide = f"{carpeta_base}/slide_lmplot_ingreso_participacion_educacion.png"

# Se puede usar 'col' o 'hue' para agrupar
g = sns.lmplot(data=df, x="income_usd", y="voter_turnout_pct", hue="education_level",
               col="education_level", col_wrap=3, height=4, aspect=1.2,
               scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
g.set_axis_labels("Ingreso Anual (USD)", "Participación Electoral (%)")
g.set_titles("Nivel Educativo: {col_name}")
g.fig.suptitle("Relación entre Ingreso y Participación Electoral por Nivel Educativo", y=1.02, fontsize=16)
plt.tight_layout()
g.fig.savefig(grafico)
guardar_slide(g.fig, objetivo, conclusion, slide) # Pasar g.fig ya que lmplot devuelve una FacetGrid
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 8: Jointplot mejorado (kde) - Edad vs Participación
# (Mejorando el jointplot original con 'kde' para mayor claridad de densidad)
# ---
titulo = "🔎 ANÁLISIS 8: Distribución Conjunta de Edad y Participación (KDE)"
objetivo = "Visualizar la densidad de la relación entre la edad y la participación electoral usando un mapa de calor y distribuciones marginales."
conclusion = "Este jointplot con KDE (Kernel Density Estimate) ofrece una vista más suave de las áreas de alta concentración, indicando dónde se cruzan las edades y los porcentajes de participación más comunes."
grafico = f"{carpeta_base}/jointplot_kde_edad_participacion.png"

joint = sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="kde", cmap="mako_r", fill=True)
joint.set_axis_labels("Edad", "Participación Electoral (%)")
joint.fig.suptitle("Distribución Conjunta de Edad y Participación Electoral", y=1.02, fontsize=16)
joint.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico) # Jointplot no usa guardar_slide directamente como fig

# ---
## Gráfico 9: Barplot mejorado - Participación promedio por Región
# (Incluyendo barras de error para incertidumbre)
# ---
titulo = "🔎 ANÁLISIS 9: Promedio de Participación Electoral por Región con Intervalo de Confianza"
objetivo = "Identificar las regiones con mayor y menor promedio de participación electoral, mostrando la variabilidad con barras de error."
conclusion = "Las barras de error en este barplot ofrecen una mejor idea de la fiabilidad del promedio de participación para cada región. Regiones con barras más cortas sugieren promedios más consistentes."
grafico = f"{carpeta_base}/barplot_participacion_region_ci.png"
slide = f"{carpeta_base}/slide_barplot_participacion_region_ci.png"

fig, ax = plt.subplots(figsize=(10, 6))
# Se usa ci='sd' para desviación estándar o ci=95 para intervalo de confianza del 95%
sns.barplot(data=df, x="voter_turnout_pct", y="region", palette="crest", ax=ax, errorbar='sd')
ax.set_title("Promedio de Participación Electoral por Región (con Desviación Estándar)", fontsize=16)
ax.set_xlabel("Participación Electoral (%)")
ax.set_ylabel("Región")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 10: Pairplot mejorado (con 'diag_kind' y 'corner')
# ---
titulo = "🔎 ANÁLISIS 10: Relaciones Multivariadas Detalladas"
objetivo = "Explorar de manera exhaustiva las relaciones bivariadas y las distribuciones univariadas entre un subconjunto de variables."
conclusion = "Este pairplot mejorado permite una exploración rápida de múltiples relaciones, identificando posibles tendencias o agrupaciones entre las variables clave y sus distribuciones. Al usar 'corner=True' se evitan gráficos duplicados."
grafico = f"{carpeta_base}/pairplot_mejorado.png"

pair = sns.pairplot(df, vars=["age", "income_usd", "voter_turnout_pct", "trust_government_pct"],
                    hue="education_level",
                    kind="reg", # Añade líneas de regresión lineal
                    diag_kind="kde", # Cambia los histogramas por KDE en la diagonal
                    corner=True, # Solo muestra la parte inferior izquierda para evitar duplicados
                    height=2.5,
                    palette="tab10")
pair.fig.suptitle("Análisis de Relaciones Multivariadas por Nivel Educativo", y=1.02, fontsize=16)
pair.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico)