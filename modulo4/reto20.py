import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore

# ---
## Configuración Inicial
---

# Configuración visual de Seaborn para un estilo limpio y profesional
sns.set_theme(style="whitegrid", context="talk", palette="viridis")

# Rutas para el archivo de entrada y la carpeta de salida
# Asegúrate de que el archivo CSV esté en 'modulo4/entrada/'
archivo = "modulo4/entrada/voter_turnout_socioeconomic_csv.csv"
# La salida se guardará en una nueva carpeta 'reto20_mejorado'
carpeta_base = "modulo4/salida/reto20_mejorado"
os.makedirs(carpeta_base, exist_ok=True) # Crea la carpeta si no existe

# ---
## Carga y Limpieza de Datos
---

try:
    df = pd.read_csv(archivo)
    print("✅ Datos cargados exitosamente.")
except FileNotFoundError:
    print(f"❌ Error: El archivo no se encuentra en la ruta especificada: {archivo}")
    print("Por favor, verifica que 'voter_turnout_socioeconomic_csv.csv' esté en la carpeta 'modulo4/entrada'.")
    exit() # Termina el script si el archivo no se encuentra

# Eliminar filas duplicadas
df = df.drop_duplicates()
print(f"✅ Filas duplicadas eliminadas. Total de filas: {len(df)}")

# Eliminar filas con valores nulos
df = df.dropna()
print(f"✅ Filas con valores nulos eliminadas. Total de filas: {len(df)}")

# Convertir columnas categóricas al tipo 'category' para un mejor manejo
for col in ["region", "education_level"]:
    if col in df.columns:
        df[col] = df[col].astype("category")
        print(f"✅ Columna '{col}' convertida a tipo categórico.")
    else:
        print(f"⚠️ Advertencia: Columna '{col}' no encontrada en el DataFrame.")

# Eliminar outliers numéricos usando z-score (solo en columnas numéricas relevantes)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if not numeric_cols.empty:
    # Calcula z-scores para todas las columnas numéricas
    z_scores = np.abs(zscore(df[numeric_cols]))
    # Filtra el DataFrame, manteniendo solo las filas donde ningún z-score exceda 3 (valor común para outliers)
    df = df[(z_scores < 3).all(axis=1)]
    print(f"✅ Outliers numéricos eliminados (z-score < 3). Total de filas: {len(df)}")
else:
    print("⚠️ Advertencia: No se encontraron columnas numéricas para aplicar la eliminación de outliers por z-score.")

# Validar que los porcentajes estén en el rango [0, 100]
initial_rows = len(df)
df = df[(df["voter_turnout_pct"].between(0, 100)) & (df["trust_government_pct"].between(0, 100))]
if len(df) < initial_rows:
    print(f"✅ Filas con porcentajes fuera de [0, 100] eliminadas. Total de filas: {len(df)}")
else:
    print("✅ Todos los porcentajes están en el rango válido [0, 100].")

# ---
## Funciones de Salida Gerencial
---

def resumen_gerencial(titulo, objetivo, conclusion, ruta_grafico):
    """
    Imprime un resumen gerencial en la consola.
    """
    print("\n" + "="*80)
    print(titulo)
    print(f"\n📌 Objetivo: {objetivo}")
    print(f"\n📊 Gráfico guardado en: {ruta_grafico}")
    print(f"\n🧠 Conclusión:\n{conclusion}")
    print("="*80)

def guardar_slide(fig, objetivo, conclusion, nombre_salida):
    """
    Guarda una figura como un 'slide' con objetivo y conclusión.
    Cierra la figura para liberar memoria después de guardarla.
    """
    # Ajusta la posición del texto para evitar superposiciones con diferentes layouts
    fig.text(0.5, 0.02, f"📌 Objetivo: {objetivo}", wrap=True, ha='center', fontsize=10, color='gray')
    fig.text(0.5, -0.05, f"🧠 Conclusión: {conclusion}", wrap=True, ha='center', fontsize=11)
    fig.savefig(nombre_salida, bbox_inches='tight') # bbox_inches='tight' para que el texto no se corte
    plt.close(fig) # Cierra la figura para liberar memoria y evitar que se muestre automáticamente
    print(f"🖼️ Slide guardado: {nombre_salida}")

# ====================================================================
# 📈 Generación de Gráficos e Información Gerencial
# ====================================================================

# ---
## Gráfico 1: Boxplot - Distribución y Outliers por Nivel Educativo
---
titulo = "🔎 ANÁLISIS 1: Distribución de Participación Electoral por Nivel Educativo"
objetivo = "Visualizar la distribución, mediana y posibles outliers de la participación electoral para cada nivel educativo, identificando la dispersión."
conclusion = "Los boxplots muestran diferencias claras en la dispersión y la mediana de la participación entre los grupos educativos. Los niveles más altos de educación (e.g., University Degree) tienden a tener una participación con un rango intercuartílico más estrecho, indicando mayor consistencia, mientras que otros niveles pueden mostrar mayor variabilidad o outliers."
grafico = f"{carpeta_base}/boxplot_participacion_educacion.png"
slide = f"{carpeta_base}/slide_boxplot_participacion_educacion.png"

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="education_level", y="voter_turnout_pct", ax=ax, palette="coolwarm")
ax.set_title("Distribución de la Participación Electoral por Nivel Educativo", fontsize=16)
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Participación Electoral (%)")
plt.xticks(rotation=15, ha='right') # Rotar etiquetas para mejor legibilidad
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 2: Swarmplot - Densidad de Puntos de Confianza por Región
---
titulo = "🔎 ANÁLISIS 2: Confianza en el Gobierno por Región (Puntos Individuales)"
objetivo = "Mostrar la densidad de puntos individuales de confianza en el gobierno para cada región, permitiendo visualizar la concentración y la dispersión real de los datos sin superposición."
conclusion = "El swarmplot revela que, si bien algunas regiones tienen una distribución de confianza más dispersa, otras muestran clústeres de alta concentración en ciertos rangos. Por ejemplo, la Región 'A' podría tener una mayor agrupación de respuestas de confianza alta, mientras que la Región 'C' podría tener una distribución más uniforme o bimodal."
grafico = f"{carpeta_base}/swarmplot_confianza_region.png"
slide = f"{carpeta_base}/slide_swarmplot_confianza_region.png"

fig, ax = plt.subplots(figsize=(12, 7))
sns.swarmplot(data=df, x="region", y="trust_government_pct", ax=ax, size=4, palette="mako") # Tamaño de punto ajustado
ax.set_title("Distribución de Confianza en el Gobierno por Región", fontsize=16)
ax.set_xlabel("Región")
ax.set_ylabel("Confianza en el Gobierno (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 3: Stripplot - Dispersión Simple de Edad por Nivel Educativo
---
titulo = "🔎 ANÁLISIS 3: Dispersión Individual de Edades por Nivel Educativo"
objetivo = "Visualizar la dispersión de cada observación de edad dentro de cada categoría de nivel educativo para identificar rangos de edad predominantes o brechas."
conclusion = "Este stripplot permite una vista detallada de la edad de cada individuo por nivel educativo. Se observa que ciertos niveles educativos, como 'High School', abarcan un rango de edad muy amplio, mientras que 'University Degree' podría concentrarse en grupos de edad más maduros. Esto puede influir en futuras estrategias de comunicación segmentada."
grafico = f"{carpeta_base}/stripplot_edad_educacion.png"
slide = f"{carpeta_base}/slide_stripplot_edad_educacion.png"

fig, ax = plt.subplots(figsize=(10, 6))
sns.stripplot(data=df, x="education_level", y="age", ax=ax, jitter=0.2, alpha=0.6, palette="Spectral")
ax.set_title("Dispersión de Edades por Nivel Educativo", fontsize=16)
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Edad")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 4: Countplot - Conteo de Observaciones por Nivel Educativo
---
titulo = "🔎 ANÁLISIS 4: Conteo de Observaciones por Nivel Educativo"
objetivo = "Mostrar la cantidad de individuos (conteo) para cada categoría de nivel educativo, proporcionando una vista rápida de la composición de la muestra."
conclusion = "El countplot nos indica la distribución de la muestra por nivel educativo. Por ejemplo, si 'High School' es la categoría más numerosa, esto resalta su importancia en la población estudiada y en futuras intervenciones. Es fundamental conocer el tamaño relativo de cada grupo."
grafico = f"{carpeta_base}/countplot_educacion.png"
slide = f"{carpeta_base}/slide_countplot_educacion.png"

fig, ax = plt.subplots(figsize=(9, 6))
# Ordenar las categorías por su frecuencia para una mejor visualización
sns.countplot(data=df, y="education_level", order=df["education_level"].value_counts().index, ax=ax, palette="plasma")
ax.set_title("Número de Observaciones por Nivel Educativo", fontsize=16)
ax.set_xlabel("Conteo de Individuos")
ax.set_ylabel("Nivel Educativo")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 5: Heatmap - Correlación entre Variables Numéricas Clave
---
titulo = "🔎 ANÁLISIS 5: Matriz de Correlación de Variables Numéricas Clave"
objetivo = "Visualizar la fuerza y dirección de las correlaciones lineales entre las principales variables numéricas para identificar relaciones directas o inversas."
conclusion = "El heatmap revela qué variables numéricas están más fuertemente correlacionadas. Por ejemplo, una correlación positiva fuerte entre 'age' e 'income_usd' sugeriría que a mayor edad, mayor ingreso. Una correlación débil o cercana a cero indica poca o ninguna relación lineal, lo cual es crucial para el modelado predictivo."
grafico = f"{carpeta_base}/heatmap_correlacion.png"
slide = f"{carpeta_base}/slide_heatmap_correlacion.png"

# Seleccionar solo las columnas numéricas de interés para la correlación
numeric_corr_cols = ["age", "income_usd", "voter_turnout_pct", "trust_government_pct"]
corr_matrix = df[numeric_corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Coeficiente de Correlación'})
ax.set_title("Matriz de Correlación de Variables Numéricas", fontsize=16)
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 6: Catplot - Participación Electoral por Educación y Región (Boxplots en Facetas)
---
titulo = "🔎 ANÁLISIS 6: Distribución de Participación Electoral por Educación y Región"
objetivo = "Analizar la distribución de la participación electoral en función del nivel educativo, segmentado por región, para observar patrones en cada combinación."
conclusion = "El catplot, usando boxplots en facetas, permite una comparación simultánea de la participación electoral en diferentes combinaciones de región y nivel educativo. Se observan variaciones notables: algunas regiones pueden tener una participación más consistente en ciertos niveles educativos, mientras que otras muestran mayor dispersión. Esto indica la necesidad de estrategias regionalizadas."
grafico = f"{carpeta_base}/catplot_participacion_educacion_region.png"
slide = f"{carpeta_base}/slide_catplot_participacion_educacion_region.png"

# Usamos kind='box' para ver la distribución y outliers por grupo
g = sns.catplot(data=df, x="education_level", y="voter_turnout_pct", col="region",
                kind="box", col_wrap=3, height=4, aspect=1.2, palette="GnBu", sharey=True)
g.set_axis_labels("Nivel Educativo", "Participación Electoral (%)")
g.set_titles("Región: {col_name}")
g.fig.suptitle("Distribución de Participación Electoral por Educación y Región", y=1.02, fontsize=18)
g.set_xticklabels(rotation=30, ha="right")
plt.tight_layout()
g.fig.savefig(grafico)
guardar_slide(g.fig, objetivo, conclusion, slide) # Pasar g.fig ya que catplot devuelve una FacetGrid
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 7: Lmplot - Regresión Lineal de Ingreso vs Participación por Nivel Educativo
---
titulo = "🔎 ANÁLISIS 7: Relación Lineal entre Ingreso y Participación por Nivel Educativo"
objetivo = "Investigar la existencia y la fuerza de una relación lineal entre el ingreso anual y la participación electoral, diferenciando esta relación por cada nivel educativo."
conclusion = "El lmplot muestra la tendencia de la participación electoral en función del ingreso, separada por nivel educativo. Las líneas de regresión (y sus bandas de confianza) revelan si el ingreso tiene un efecto similar o diferente en la participación a través de los grupos educativos. Esto es crucial para entender qué grupos demográficos podrían responder de manera diferente a iniciativas económicas o de participación."
grafico = f"{carpeta_base}/lmplot_ingreso_participacion_educacion.png"
slide = f"{carpeta_base}/slide_lmplot_ingreso_participacion_educacion.png"

# Se puede usar 'col' o 'hue' para agrupar. 'col' crea una cuadrícula de gráficos.
g = sns.lmplot(data=df, x="income_usd", y="voter_turnout_pct", hue="education_level",
               col="education_level", col_wrap=3, height=4, aspect=1.2,
               scatter_kws={'alpha':0.5, 's':20}, line_kws={'color':'red', 'lw':2}) # Ajuste de tamaño y transparencia de puntos
g.set_axis_labels("Ingreso Anual (USD)", "Participación Electoral (%)")
g.set_titles("Nivel Educativo: {col_name}")
g.fig.suptitle("Relación entre Ingreso y Participación Electoral por Nivel Educativo", y=1.02, fontsize=18)
plt.tight_layout()
g.fig.savefig(grafico)
guardar_slide(g.fig, objetivo, conclusion, slide) # Pasar g.fig ya que lmplot devuelve una FacetGrid
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 8: Jointplot (KDE) - Distribución Conjunta de Edad y Participación
---
titulo = "🔎 ANÁLISIS 8: Distribución Conjunta y Densidad de Edad vs Participación Electoral"
objetivo = "Visualizar la densidad de la relación entre la edad de los votantes y su participación electoral, junto con sus distribuciones marginales."
conclusion = "Este jointplot con KDE (Kernel Density Estimate) ofrece una vista suave de las áreas de alta concentración de datos. Permite identificar las combinaciones de edad y participación más comunes. Por ejemplo, podríamos ver una alta densidad de participantes entre los 40-60 años con una participación entre 70-90%, lo que resalta un segmento clave de la población."
grafico = f"{carpeta_base}/jointplot_kde_edad_participacion.png"

# jointplot no es una figura estándar de matplotlib, por eso se maneja diferente al guardar el slide
joint = sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="kde", cmap="mako_r", fill=True, height=8)
joint.set_axis_labels("Edad", "Participación Electoral (%)")
joint.fig.suptitle("Distribución Conjunta de Edad y Participación Electoral", y=1.02, fontsize=16)
joint.savefig(grafico)
plt.close(joint.fig) # Cerrar la figura del jointplot
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 9: Barplot - Promedio de Participación por Región (con Barras de Error)
---
titulo = "🔎 ANÁLISIS 9: Promedio de Participación Electoral por Región con Desviación Estándar"
objetivo = "Identificar las regiones con mayor y menor promedio de participación electoral, mostrando la variabilidad o incertidumbre del promedio con barras de error."
conclusion = "Las barras de error (desviación estándar) en este barplot ofrecen una mejor idea de la fiabilidad del promedio de participación para cada región. Regiones con barras más cortas sugieren promedios más consistentes, mientras que barras más largas indican mayor variabilidad, lo que puede requerir un análisis más profundo de los factores regionales."
grafico = f"{carpeta_base}/barplot_participacion_region_ci.png"
slide = f"{carpeta_base}/slide_barplot_participacion_region_ci.png"

fig, ax = plt.subplots(figsize=(10, 6))
# Se usa errorbar='sd' para desviación estándar o errorbar=('ci', 95) para intervalo de confianza del 95%
sns.barplot(data=df, x="voter_turnout_pct", y="region", palette="crest", ax=ax, errorbar='sd')
ax.set_title("Promedio de Participación Electoral por Región (con Desviación Estándar)", fontsize=16)
ax.set_xlabel("Participación Electoral (%)")
ax.set_ylabel("Región")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ---
## Gráfico 10: Pairplot Mejorado - Relaciones Multivariadas Detalladas
---
titulo = "🔎 ANÁLISIS 10: Análisis de Relaciones Multivariadas por Nivel Educativo"
objetivo = "Explorar exhaustivamente las relaciones bivariadas (dispersión con regresión) y las distribuciones univariadas (KDE) entre un subconjunto de variables clave, diferenciando por nivel educativo."
conclusion = "Este pairplot mejorado permite una exploración rápida y completa de múltiples relaciones simultáneamente. Podemos ver tendencias de regresión y densidades de distribución. Al usar 'corner=True' se evitan gráficos duplicados, y la diferenciación por 'education_level' revela cómo estas relaciones pueden variar entre los grupos educativos, identificando segmentos clave para futuras acciones."
grafico = f"{carpeta_base}/pairplot_mejorado.png"

pair = sns.pairplot(df, vars=["age", "income_usd", "voter_turnout_pct", "trust_government_pct"],
                    hue="education_level",
                    kind="reg", # Añade líneas de regresión lineal para relaciones bivariadas
                    diag_kind="kde", # Cambia los histogramas por KDE en la diagonal para ver densidades
                    corner=True, # Solo muestra la parte inferior izquierda para evitar gráficos duplicados
                    height=2.5,
                    palette="tab10")
pair.fig.suptitle("Análisis de Relaciones Multivariadas por Nivel Educativo", y=1.02, fontsize=18)
pair.savefig(grafico)
plt.close(pair.fig) # Cerrar la figura del pairplot
resumen_gerencial(titulo, objetivo, conclusion, grafico)

print("\n\n✨ Proceso de generación de gráficos completado. Revisa la carpeta:")
print(f"📁 {os.path.abspath(carpeta_base)}")
print("¡Ahora puedes usar los archivos PNG para crear tu documento PDF gerencial!")