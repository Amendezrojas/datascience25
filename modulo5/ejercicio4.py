#Ejercicio 4 –Efecto del diseño de una app en la rapidez de compra
# Contexto:Una empresa de e-commerce está probando dos versiones de su app (A y B).
# Desean saber si una versión permite completar la compra más rápido.
#1) Crear el df
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
 
# 2. Crearl el cojnunto de datos simulados
 
np.random.seed(42)  # Para reproducibilidad
n = 150  # Número de participantes
#simular puntuaciones de memoria con música y sin música
app_A = np.random.normal(loc=75, scale=10, size=n).round(2)
app_B = np.random.normal(loc=60, scale=5, size=n).round(2)
# Crear un DataFrame
datos = pd.DataFrame({
    'Cliente': [f"Cliente{i+1}" for i in range(n)] * 2, # "M1", "M2", ..., "M15"
    "APP": ['app_A'] * n + ['app_B'] * n,
    'Puntuación': np.concatenate([app_A, app_B])
})
 
# Mostrar las primeras filas del DataFrame
print(datos.head())
datos.to_csv('entrada/ensayo_app.csv', index=False)
print(datos.sample(20))
 
estadisticas_descriptivas = datos.describe()
resumen = datos.groupby('APP')['Puntuación'].describe()
print (resumen)
 
# Paso 7: Visualizar los datos
plt.figure(figsize=(10, 6))
datos.boxplot(column='Puntuación', by='APP', grid=False)
plt.title('Puntuaciones de satisfacción')
plt.suptitle('')
plt.xlabel('APP')
plt.ylabel('Puntuación de satisfacción')
plt.tight_layout()
plt.savefig('salida/boxplot_musica_memoria.png')
plt.show()
 
#interpretar los resultados
alpha = 0.05  # Nivel de significancia
grupo_con_musica = datos[datos['APP'] == 'APP_A']['Puntuación']
grupo_sin_musica = datos[datos['APP'] == 'APP_B']['Puntuación']
pval = stats.ttest_ind(grupo_con_musica, grupo_sin_musica, equal_var=False)
print(f"p-valor: {pval.pvalue:.4f}")
if pval.pvalue < alpha:
    print("Rechazamos la hipótesis nula: La música tiene un efecto significativo en las puntuaciones de memoria.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente para afirmar que la música afecta las puntuaciones de memoria.")
 