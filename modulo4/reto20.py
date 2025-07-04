import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Estilo visual
sns.set_theme(style="whitegrid", context="talk")

# Rutas
archivo = "modulo4/entrada/voter_turnout_socioeconomic.csv"
carpeta_salida = "modulo4/salida/graficas"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar datos
df = pd.read_csv(archivo)
print("Primeras filas:", df.head())

# Gráfico 1: Distribución de participación electoral por edad (histplot + kde)
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=30, kde=True, color="skyblue")
plt.title("Distribución de Edad de Votantes", fontsize=15)
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/histplot_age.png")
plt.show()

# Gráfico 2: KDEplot - Participación electoral vs ingresos
plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x="income_usd", y="voter_turnout_pct", fill=True, cmap="viridis")
plt.title("Participación Electoral vs Ingreso")
plt.xlabel("Ingreso (USD)")
plt.ylabel("Participación Electoral (%)")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/kde_income_turnout.png")
plt.show()

# Gráfico 3: Jointplot - Edad vs Participación Electoral
sns.jointplot(data=df, x="age", y="voter_turnout_pct", kind="hex", cmap="Blues")
plt.savefig(f"{carpeta_salida}/jointplot_age_turnout.png")
plt.show()

# Gráfico 4: Violinplot - Confianza en el gobierno por nivel educativo
plt.figure(figsize=(10,5))
sns.violinplot(data=df, x="education_level", y="trust_government_pct", palette="pastel", inner="box")
plt.title("Confianza en el Gobierno según Nivel Educativo")
plt.xlabel("Nivel Educativo")
plt.ylabel("Confianza en el Gobierno (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/violin_educ_trustgov.png")
plt.show()

# Gráfico 5: Barplot - Promedio de participación electoral por región
plt.figure(figsize=(10,6))
region_avg = df.groupby("region", as_index=False)["voter_turnout_pct"].mean().sort_values(by="voter_turnout_pct", ascending=False)
sns.barplot(data=region_avg, x="voter_turnout_pct", y="region", palette="mako")
plt.title("Promedio de Participación Electoral por Región")
plt.xlabel("Participación Electoral (%)")
plt.ylabel("Región")
plt.tight_layout()
plt.savefig(f"{carpeta_salida}/barplot_region_turnout.png")
plt.show()

# Gráfico 6: Pairplot - Relación entre variables principales coloreado por educación
sns.pairplot(df, vars=["voter_turnout_pct", "income_usd", "trust_government_pct"], hue="education_level", height=2.5)
plt.suptitle("Relaciones entre participación, ingresos y confianza", fontsize=16, y=1.02)
plt.savefig(f"{carpeta_salida}/pairplot_turnout_income_trust.png")
plt.show()

# Gráfico 7: FacetGrid - Participación electoral por educación y región
g = sns.FacetGrid(df, col="region", row="education_level", margin_titles=True, height=2.8)
g.map(sns.histplot, "voter_turnout_pct", bins=15, color="salmon")
g.set_axis_labels("Participación Electoral (%)", "Frecuencia")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribución de Participación por Región y Educación")
g.savefig(f"{carpeta_salida}/facetgrid_region_educ_turnout.png")
plt.show()
