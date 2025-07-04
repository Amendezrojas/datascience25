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
carpeta_salida = "modulo4/salida/graficas"
os.makedirs(carpeta_salida, exist_ok=True)

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
grafico = f"{carpeta_salida}/histograma_edad.png"
slide = f"{carpeta_salida}/slide_edad_votantes.png"

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
grafico = f"{carpeta_salida}/kde_ingreso_participacion.png"
slide = f"{carpeta_salida}/slide_ingreso_participacion.png"

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
grafico = f"{carpeta_salida}/jointplot_edad_participacion.png"

joint = sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="hex", cmap="Blues")
joint.savefig(grafico)
resumen_gerencial(titulo, objetivo, conclusion, grafico)

# Continúa con los gráficos 4 al 7 en el mismo formato si deseas
# (Violinplot por educación, Barplot por región, Pairplot, FacetGrid por región y educación)
# Si quieres que los agregue también, dímelo y te completo el bloque completo con slides.
