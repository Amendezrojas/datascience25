import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore

# Configuración visual de los gráficos
sns.set_theme(style="whitegrid", context="talk")

# Definición de rutas
archivo = "modulo4/entrada/voter_turnout_socioeconomic_csv.csv"
carpeta_salida = "modulo4/salida/graficas"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar el dataset
df = pd.read_csv(archivo)
print("🔍 Primeras filas del conjunto de datos:")
print(df.head())
print(f"\n📐 Dimensiones iniciales: {df.shape[0]} filas y {df.shape[1]} columnas")

# =======================
# 🧼 LIMPIEZA DE DATOS
# =======================

# 1. Duplicados
duplicados = df.duplicated().sum()
print(f"\n🧽 Duplicados encontrados: {duplicados}")
df = df.drop_duplicates()

# 2. Columnas totalmente nulas
columnas_nulas_totales = df.columns[df.isnull().all()]
if len(columnas_nulas_totales) > 0:
    print(f"❌ Columnas totalmente nulas eliminadas: {list(columnas_nulas_totales)}")
    df = df.drop(columns=columnas_nulas_totales)

# 3. Valores nulos por columna
print("\n📊 Valores nulos por columna:")
print(df.isnull().sum())

# Eliminamos filas con valores nulos (o podrías imputar si prefieres)
df = df.dropna()

# 4. Convertimos columnas categóricas
columnas_categoricas = ["region", "education_level"]
for col in columnas_categoricas:
    if col in df.columns:
        df[col] = df[col].astype("category")

# 5. Eliminación de outliers (z-score)
columnas_numericas = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(columnas_numericas))
df = df[(z_scores < 3).all(axis=1)]

# 6. Validamos rango válido para porcentajes
df = df[
    (df["voter_turnout_pct"].between(0, 100)) &
    (df["trust_government_pct"].between(0, 100))
]

print(f"\n✅ Datos limpios. Dimensiones finales: {df.shape[0]} filas y {df.shape[1]} columnas")

# =======================
# 📊 VISUALIZACIONES
# =======================

# Histograma de edad
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True, color="skyblue")
plt.title("Distribución de Edad de los Votantes")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/histograma_edad.png")
plt.show()

# KDEplot: participación electoral vs ingreso
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="income_usd", y="voter_turnout_pct", fill=True, cmap="viridis")
plt.title("Participación Electoral vs Ingreso Anual")
plt.xlabel("Ingreso Anual (USD)")
plt.ylabel("Participación Electoral (%)")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/kde_ingreso_participacion.png")
plt.show()

# Jointplot: edad vs participación electoral
sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="hex", cmap="Blues")
plt.savefig(f"{carpeta_salida}/jointplot_edad_participacion.png")
plt.show()

# Violinplot: confianza en el gobierno por nivel educativo
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x="education_level", y="trust_government_pct", palette="pastel", inner="box")
plt.title("Confianza en el Gobierno según Nivel Educativo")
plt.xlabel("Nivel Educativo")
plt.ylabel("Confianza en el Gobierno (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/violin_educacion_confianza.png")
plt.show()

# Barplot: promedio de participación electoral por región
plt.figure(figsize=(10, 6))
region_avg = df.groupby("region", as_index=False)["voter_turnout_pct"].mean().sort_values(by="voter_turnout_pct", ascending=False)
sns.barplot(data=region_avg, x="voter_turnout_pct", y="region", palette="mako")
plt.title("Promedio de Participación Electoral por Región")
plt.xlabel("Participación Electoral (%)")
plt.ylabel("Región")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/barplot_participacion_region.png")
plt.show()

# Pairplot: relación entre participación, ingresos y confianza (color según educación)
sns.pairplot(df, vars=["voter_turnout_pct", "income_usd", "trust_government_pct"], hue="education_level", height=2.5)
plt.suptitle("Relaciones entre Participación, Ingresos y Confianza", fontsize=16, y=1.02)
plt.savefig(f"{carpeta_salida}/pairplot_participacion_ingreso_confianza.png")
plt.show()

# FacetGrid: participación electoral por región y educación
g = sns.FacetGrid(df, col="region", row="education_level", margin_titles=True, height=2.8)
g.map(sns.histplot, "voter_turnout_pct", bins=15, color="salmon")
g.set_axis_labels("Participación Electoral (%)", "Frecuencia")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribución de Participación por Región y Nivel Educativo")
g.savefig(f"{carpeta_salida}/facetgrid_participacion_region_educacion.png")
plt.show()
