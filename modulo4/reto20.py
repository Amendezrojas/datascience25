import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore

# Configuración visual
sns.set_theme(style="whitegrid", context="talk")

# Rutas
archivo = "modulo4/entrada/voter_turnout_socioeconomic_csv.csv"
carpeta_base = "modulo4/salida/reto20"
os.makedirs(carpeta_base, exist_ok=True)

# ========================
# 🔹 CARGA Y LIMPIEZA DE DATOS
# ========================
df = pd.read_csv(archivo)
df = df.drop_duplicates()
df = df.dropna()

# Convertir columnas categóricas
for col in ["region", "education_level"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Eliminar outliers numéricos usando z-score
z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
df = df[(z_scores < 3).all(axis=1)]

# Validar porcentajes en rango [0, 100]
df = df[(df["voter_turnout_pct"].between(0, 100)) & (df["trust_government_pct"].between(0, 100))]

# ========================
# 🔹 FUNCIONES DE SALIDA GERENCIAL
# ========================
def resumen_gerencial(titulo, objetivo, conclusion, ruta_grafico):
    print("\n" + "="*80)
    print(titulo)
    print(f"\n📌 Objetivo: {objetivo}")
    print(f"\n📊 Gráfico guardado en: {ruta_grafico}")
    print(f"\n🧠 Conclusión:\n{conclusion}")
    print("="*80)

def guardar_slide(fig, objetivo, conclusion, nombre_salida):
    plt.figtext(0.5, -0.08, f"📌 Objetivo: {objetivo}", wrap=True, ha='center', fontsize=10, color='gray')
    plt.figtext(0.5, -0.13, f"🧠 Conclusión: {conclusion}", wrap=True, ha='center', fontsize=11)
    fig.savefig(nombre_salida, bbox_inches='tight')
    plt.show()

# ========================
# 📈 GRÁFICO 1: Histograma Edad
# ========================
titulo = "🔎 ANÁLISIS 1: ¿Cómo está distribuida la edad de los votantes?"
objetivo = "Observar la distribución de edades para entender el perfil etario de los votantes."
conclusion = "La mayoría de los votantes se concentra entre los 30 y 50 años, con menor presencia en jóvenes y adultos mayores."
grafico = f"{carpeta_base}/histograma_edad.png"
slide = f"{carpeta_base}/slide_edad_votantes.png"

fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(df["age"], bins=30, kde=True, color="skyblue", ax=ax)
ax.set_title("Distribución de Edad de los Votantes", fontsize=16)
ax.set_xlabel("Edad")
ax.set_ylabel("Frecuencia")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 2: KDE ingreso vs participación
# ========================
titulo = "🔎 ANÁLISIS 2: ¿Cómo se relacionan los ingresos con la participación electoral?"
objetivo = "Explorar si hay una correlación entre ingreso anual y nivel de participación."
conclusion = "Se observa una leve tendencia a mayor participación en tramos medios de ingreso, pero no es concluyente."
grafico = f"{carpeta_base}/kde_ingreso_participacion.png"
slide = f"{carpeta_base}/slide_ingreso_participacion.png"

fig, ax = plt.subplots(figsize=(9, 6))
sns.kdeplot(data=df, x="income_usd", y="voter_turnout_pct", fill=True, cmap="viridis", ax=ax)
ax.set_title("Participación Electoral vs Ingreso Anual", fontsize=16)
ax.set_xlabel("Ingreso Anual (USD)")
ax.set_ylabel("Participación Electoral (%)")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 3: Jointplot edad vs participación
# ========================
titulo = "🔎 ANÁLISIS 3: ¿Qué relación existe entre la edad y la participación?"
objetivo = "Visualizar si hay relación entre la edad de los votantes y su participación electoral."
conclusion = "Se aprecia una mayor densidad de participación entre los 30 y 60 años, aunque con bastante dispersión."
grafico = f"{carpeta_base}/jointplot_edad_participacion.png"

joint = sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="hex", cmap="Blues")
joint.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 4: Violinplot confianza vs educación
# ========================
titulo = "🔎 ANÁLISIS 4: ¿Cómo varía la confianza en el gobierno según el nivel educativo?"
objetivo = "Analizar si el nivel educativo influye en la confianza depositada en el gobierno."
conclusion = "Se observan diferencias claras: los niveles más altos de educación tienden a tener mayor variabilidad y menor confianza promedio."
grafico = f"{carpeta_base}/violin_educacion_confianza.png"
slide = f"{carpeta_base}/slide_educacion_confianza.png"

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df, x="education_level", y="trust_government_pct", palette="pastel", inner="box", ax=ax)
ax.set_title("Confianza en el Gobierno según Nivel Educativo")
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Confianza en el Gobierno (%)")
plt.xticks(rotation=30)
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 5: Barplot participación por región
# ========================
titulo = "🔎 ANÁLISIS 5: ¿Qué regiones tienen mayor participación electoral promedio?"
objetivo = "Identificar las regiones con mayor y menor promedio de participación electoral."
conclusion = "Algunas regiones destacan por sobre otras, lo que podría asociarse a factores culturales o socioeconómicos."
grafico = f"{carpeta_base}/barplot_participacion_region.png"
slide = f"{carpeta_base}/slide_participacion_region.png"

fig, ax = plt.subplots(figsize=(10, 6))
region_avg = df.groupby("region", as_index=False)["voter_turnout_pct"].mean().sort_values(by="voter_turnout_pct", ascending=False)
sns.barplot(data=region_avg, x="voter_turnout_pct", y="region", palette="mako", ax=ax)
ax.set_title("Promedio de Participación Electoral por Región")
ax.set_xlabel("Participación Electoral (%)")
ax.set_ylabel("Región")
plt.tight_layout()
fig.savefig(grafico)
guardar_slide(fig, objetivo, conclusion, slide)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 6: Pairplot
# ========================
titulo = "🔎 ANÁLISIS 6: Relaciones entre participación, ingreso y confianza"
objetivo = "Explorar posibles relaciones entre participación electoral, ingreso y confianza en el gobierno."
conclusion = "La participación y la confianza parecen no tener una relación directa clara; el ingreso muestra leve asociación."
grafico = f"{carpeta_base}/pairplot_participacion_ingreso_confianza.png"

pair = sns.pairplot(df, vars=["voter_turnout_pct", "income_usd", "trust_government_pct"], hue="education_level", height=2.5)
pair.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# ========================
# 📈 GRÁFICO 7: FacetGrid por región y educación
# ========================
titulo = "🔎 ANÁLISIS 7: Participación electoral segmentada por región y educación"
objetivo = "Observar la distribución de participación electoral cruzando región y nivel educativo."
conclusion = "Se observan diferencias notables en participación dependiendo de la combinación de región y educación."
grafico = f"{carpeta_base}/facetgrid_participacion_region_educacion.png"

g = sns.FacetGrid(df, col="region", row="education_level", margin_titles=True, height=2.8)
g.map(sns.histplot, "voter_turnout_pct", bins=15, color="salmon")
g.set_axis_labels("Participación Electoral (%)", "Frecuencia")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribución de Participación por Región y Nivel Educativo")
g.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico)
