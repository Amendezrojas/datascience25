import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, poisson, mode
import os # Importar el módulo os para la creación de directorios

# --- RETO 1: Resistencia de Vigas de Concreto 🏗️ ---
print("--- INICIO RETO 1: Resistencia de Vigas de Concreto ---")

# Crear carpeta de salida para Reto 1
output_dir_r1 = 'salida_reto_1'
os.makedirs(output_dir_r1, exist_ok=True)

# 1. Simular 250 valores de resistencia con distribución normal (media 28, desviación 3)
np.random.seed(42) # Para reproducibilidad
media_resistencia = 28 # MPa
desviacion_resistencia = 3 # MPa
num_ensayos = 250
resistencia_simulada = np.random.normal(loc=media_resistencia, scale=desviacion_resistencia, size=num_ensayos)

print("\nPrimeros 10 valores de resistencia simulada (Reto 1):")
print(resistencia_simulada[:10].round(2))

# 2. Crear un histograma con curva KDE para visualizar la distribución
plt.figure(figsize=(10, 6))
sns.histplot(resistencia_simulada, kde=True, bins=20, stat='density', color='skyblue', label='Distribución Simulada')
plt.title('Reto 1: Histograma de Resistencia a la Compresión con KDE')
plt.xlabel('Resistencia (MPa)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r1, 'reto1_histograma_resistencia.png')) # Guardar figura
plt.show()

# 3. Graficar la función de densidad de la distribución normal (PDF teórica)
x_pdf_r1 = np.linspace(resistencia_simulada.min(), resistencia_simulada.max(), 500)
pdf_teorica_r1 = norm.pdf(x_pdf_r1, loc=media_resistencia, scale=desviacion_resistencia)

plt.figure(figsize=(10, 6))
plt.plot(x_pdf_r1, pdf_teorica_r1, color='red', linestyle='--', label='PDF Teórica (Normal)')
sns.histplot(resistencia_simulada, kde=True, bins=20, stat='density', color='skyblue', alpha=0.7, label='Histograma Simulado')
plt.title('Reto 1: PDF Teórica vs. Histograma de Resistencia')
plt.xlabel('Resistencia (MPa)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r1, 'reto1_pdf_vs_histograma.png')) # Guardar figura
plt.show()

# 4. Calcular la probabilidad de que una viga tenga resistencia menor a 25 MPa
valor_critico_r1 = 25 # MPa
prob_menor_25_r1 = norm.cdf(valor_critico_r1, loc=media_resistencia, scale=desviacion_resistencia)
print(f"\nReto 1: Probabilidad de que una viga tenga resistencia menor a {valor_critico_r1} MPa: {prob_menor_25_r1:.4f}")

# 5. Sombrear el área bajo la curva para representar esa probabilidad
plt.figure(figsize=(10, 6))
plt.plot(x_pdf_r1, pdf_teorica_r1, color='red', linestyle='--', label='PDF Teórica (Normal)')
x_sombreado_r1 = np.linspace(x_pdf_r1.min(), valor_critico_r1, 100)
plt.fill_between(x_sombreado_r1, 0, norm.pdf(x_sombreado_r1, loc=media_resistencia, scale=desviacion_resistencia),
                 color='orange', alpha=0.5, label=f'P(Resistencia < {valor_critico_r1} MPa)')
plt.axvline(x=valor_critico_r1, color='blue', linestyle=':', label=f'{valor_critico_r1} MPa')
plt.title(f'Reto 1: Probabilidad de Resistencia Menor a {valor_critico_r1} MPa')
plt.xlabel('Resistencia (MPa)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r1, 'reto1_prob_menor_25.png')) # Guardar figura
plt.show()

# 6. Comentar si la proporción de vigas por debajo de 25 MPa representa un riesgo para la obra
print("\nReto 1: Comentario sobre el riesgo para la obra:")
if prob_menor_25_r1 > 0.05:
    print(f"La probabilidad de que una viga tenga resistencia menor a {valor_critico_r1} MPa es del {prob_menor_25_r1:.2%}.")
    print("Esta proporción puede representar un riesgo significativo para la obra, ya que una resistencia baja podría")
    print("comprometer la integridad estructural. Se debería investigar la causa de estas resistencias bajas y")
    print("considerar ajustes en el proceso de mezclado, curado o materiales para garantizar la seguridad y calidad.")
else:
    print(f"La probabilidad de que una viga tenga resistencia menor a {valor_critico_r1} MPa es del {prob_menor_25_r1:.2%}.")
    print("Esta proporción es relativamente baja, lo que sugiere que el riesgo asociado a una baja resistencia es")
    print("manejable. Sin embargo, el monitoreo continuo y la búsqueda de mejoras en el proceso siempre son recomendables.")
print("--- FIN RETO 1 ---")

# --- RETO 2: Número de Fallas Técnicas por Semana 👷‍♀️ ---
print("\n--- INICIO RETO 2: Número de Fallas Técnicas por Semana ---")

# Crear carpeta de salida para Reto 2
output_dir_r2 = 'salida_reto_2'
os.makedirs(output_dir_r2, exist_ok=True)

# Parámetro lambda para la distribución de Poisson (promedio de fallas por semana)
lamda_r2 = 2

# 1. Simular el número de fallas semanales durante 50 semanas usando una distribución de Poisson con lambda=2
np.random.seed(42) # Para reproducibilidad
num_semanas_r2 = 50
fallas_semanales_simuladas_r2 = poisson.rvs(mu=lamda_r2, size=num_semanas_r2)

print("\nPrimeros 10 valores de fallas semanales simuladas (Reto 2):")
print(fallas_semanales_simuladas_r2[:10])

# 2. Crear una tabla con la frecuencia de cada cantidad de fallas
frecuencia_fallas_r2 = pd.Series(fallas_semanales_simuladas_r2).value_counts().sort_index()
print("\nReto 2: Tabla de Frecuencia de Fallas Técnicas Semanales:")
print(frecuencia_fallas_r2)

# 3. Graficar la distribución usando un gráfico de barras
plt.figure(figsize=(10, 6))
frecuencia_fallas_r2.plot(kind='bar', color='lightcoral')
plt.title('Reto 2: Distribución de Fallas Técnicas Semanales (50 Semanas Simuladas)')
plt.xlabel('Número de Fallas')
plt.ylabel('Frecuencia (Número de Semanas)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir_r2, 'reto2_distribucion_fallas.png')) # Guardar figura
plt.show()

# 4. Calcular la probabilidad de tener exactamente 3 fallas en una semana
k_fallas_r2 = 3
prob_exactamente_3_fallas_r2 = poisson.pmf(k=k_fallas_r2, mu=lamda_r2)
print(f"\nReto 2: Probabilidad de tener exactamente {k_fallas_r2} fallas en una semana: {prob_exactamente_3_fallas_r2:.4f}")

# 5. Interpretar el resultado: ¿es raro tener más de 3 fallas?, ¿es preocupante?
prob_mas_de_3_fallas_r2 = 1 - poisson.cdf(k=k_fallas_r2, mu=lamda_r2)
print(f"Reto 2: Probabilidad de tener más de {k_fallas_r2} fallas en una semana: {prob_mas_de_3_fallas_r2:.4f}")

print("\nReto 2: Interpretación del resultado:")
if prob_exactamente_3_fallas_r2 > 0.15:
    print(f"- La probabilidad de tener exactamente {k_fallas_r2} fallas en una semana ({prob_exactamente_3_fallas_r2:.2%}) es relativamente común.")
else:
    print(f"- La probabilidad de tener exactamente {k_fallas_r2} fallas en una semana ({prob_exactamente_3_fallas_r2:.2%}) no es tan alta.")

if prob_mas_de_3_fallas_r2 < 0.05:
    print(f"- Es raro tener más de {k_fallas_r2} fallas en una semana (probabilidad: {prob_mas_de_3_fallas_r2:.2%}).")
    print("- No es tan preocupante si ocurre ocasionalmente, pero se debe monitorear la tendencia.")
else:
    print(f"- La probabilidad de tener más de {k_fallas_r2} fallas en una semana ({prob_mas_de_3_fallas_r2:.2%}) es considerable.")
    print("- Esto podría ser preocupante y requerir atención para identificar las causas y reducir la frecuencia de fallas mayores.")

# 6. Sugerir una mejora para prevenir las fallas más frecuentes
print("\nReto 2: Sugerencia de mejora para prevenir las fallas más frecuentes:")
print("- Dada la distribución, las cantidades de 0, 1, 2 y 3 fallas son las más probables.")
print("- Se podría implementar un **programa de mantenimiento preventivo más riguroso y basado en datos** para los equipos críticos,")
print("  realizando inspecciones periódicas y reparaciones proactivas antes de que las fallas ocurran.")
print("- Además, la **capacitación continua del personal** en el uso correcto de la maquinaria y en los procedimientos de seguridad")
print("  puede reducir significativamente los errores operativos que conducen a fallas.")
print("- Establecer un **sistema de reporte y análisis de fallas** para identificar patrones y causas raíz recurrentes.")
print("--- FIN RETO 2 ---")

# --- RETO 3: Tiempo de Carga de Material ⏳ ---
print("\n--- INICIO RETO 3: Tiempo de Carga de Material ---")

# Crear carpeta de salida para Reto 3
output_dir_r3 = 'salida_reto_3'
os.makedirs(output_dir_r3, exist_ok=True)

# Parámetros de la distribución del tiempo de carga
media_carga_r3 = 12 # minutos
desviacion_carga_r3 = 2.5 # minutos

# 1. Simular 100 tiempos de carga usando np.random.normal()
np.random.seed(42) # Para reproducibilidad
num_simulaciones_r3 = 100
tiempos_carga_simulados_r3 = np.random.normal(loc=media_carga_r3, scale=desviacion_carga_r3, size=num_simulaciones_r3)

print("\nPrimeros 10 tiempos de carga simulados (Reto 3):")
print(tiempos_carga_simulados_r3[:10].round(2))

# 2. Ordenar los datos y construir la CDF empírica (distribución acumulada)
tiempos_carga_ordenados_r3 = np.sort(tiempos_carga_simulados_r3)
cdf_empirica_r3 = np.arange(1, len(tiempos_carga_ordenados_r3) + 1) / len(tiempos_carga_ordenados_r3)

# 3. Graficar la curva CDF (empírica y teórica)
plt.figure(figsize=(10, 6))
plt.plot(tiempos_carga_ordenados_r3, cdf_empirica_r3, label='CDF Empírica (Datos Simulados)', color='blue')

# Generar la CDF Teórica
x_cdf_r3 = np.linspace(tiempos_carga_ordenados_r3.min(), tiempos_carga_ordenados_r3.max(), 500)
cdf_teorica_r3 = norm.cdf(x_cdf_r3, loc=media_carga_r3, scale=desviacion_carga_r3)
plt.plot(x_cdf_r3, cdf_teorica_r3, label='CDF Teórica (Normal)', color='red', linestyle='--')

plt.title('Reto 3: Función de Distribución Acumulada (CDF) del Tiempo de Carga')
plt.xlabel('Duración de la Carga (minutos)')
plt.ylabel('Probabilidad Acumulada')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r3, 'reto3_cdf_tiempo_carga.png')) # Guardar figura
plt.show()

# 4. Calcular la probabilidad de que una carga tarde más de 15 minutos usando la CDF teórica
valor_critico_retraso_r3 = 15 # minutos
prob_mas_de_15_min_r3 = 1 - norm.cdf(valor_critico_retraso_r3, loc=media_carga_r3, scale=desviacion_carga_r3)
print(f"\nReto 3: Probabilidad de que una carga tarde más de {valor_critico_retraso_r3} minutos: {prob_mas_de_15_min_r3:.4f}")

# 5. Marcar ese punto en la gráfica
plt.figure(figsize=(10, 6))
plt.plot(tiempos_carga_ordenados_r3, cdf_empirica_r3, label='CDF Empírica (Datos Simulados)', color='blue')
plt.plot(x_cdf_r3, cdf_teorica_r3, label='CDF Teórica (Normal)', color='red', linestyle='--')

prob_hasta_15_min_r3 = norm.cdf(valor_critico_retraso_r3, loc=media_carga_r3, scale=desviacion_carga_r3)
plt.plot(valor_critico_retraso_r3, prob_hasta_15_min_r3, 'o', markersize=8, color='green',
         label=f'P(X <= {valor_critico_retraso_r3} min) = {prob_hasta_15_min_r3:.2f}')
plt.axvline(x=valor_critico_retraso_r3, color='green', linestyle=':', label=f'{valor_critico_retraso_r3} minutos')
plt.axhline(y=prob_hasta_15_min_r3, color='green', linestyle=':', label=f'Prob. Acumulada en {valor_critico_retraso_r3} min')

plt.title(f'Reto 3: CDF del Tiempo de Carga con Probabilidad de >{valor_critico_retraso_r3} minutos')
plt.xlabel('Duración de la Carga (minutos)')
plt.ylabel('Probabilidad Acumulada')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r3, 'reto3_cdf_prob_retraso.png')) # Guardar figura
plt.show()

# 6. Discutir si se deben usar más grúas o ajustar turnos
print("\nReto 3: Discusión sobre el ajuste de operaciones:")
if prob_mas_de_15_min_r3 > 0.10:
    print(f"La probabilidad de que una carga tarde más de {valor_critico_retraso_r3} minutos es del {prob_mas_de_15_min_r3:.2%}.")
    print("Esto indica que los retrasos significativos son relativamente frecuentes. Si estos retrasos afectan")
    print("la cadena logística o los plazos del proyecto, se debería considerar:")
    print("  - **Usar más grúas**: Si la capacidad de carga es un cuello de botella, aumentar el número de grúas")
    print("    disponibles podría reducir los tiempos de espera y agilizar el proceso.")
    print("  - **Ajustar los turnos**: Redistribuir la carga de trabajo o ajustar los turnos del personal para evitar")
    print("    picos de demanda y asegurar que haya suficiente personal y equipo durante los momentos de mayor actividad.")
    print("  - **Optimización de procesos**: Analizar las causas específicas de los tiempos de carga más largos (ej. ineficiencias,")
    print("    problemas de coordinación, mantenimiento de equipos) para implementar mejoras operativas.")
else:
    print(f"La probabilidad de que una carga tarde más de {valor_critico_retraso_r3} minutos es del {prob_mas_de_15_min_r3:.2%}.")
    print("Esta probabilidad es relativamente baja. Aunque siempre hay margen para la mejora, los retrasos significativos")
    print("no parecen ser un problema recurrente que requiera cambios drásticos inmediatos. Se podría monitorear la situación")
    print("y buscar optimizar pequeñas ineficiencias.")
print("--- FIN RETO 3 ---")

# --- RETO 4: Conteo Vehicular en un Puente 🚗 ---
print("\n--- INICIO RETO 4: Conteo Vehicular en un Puente ---")

# Crear carpeta de salida para Reto 4
output_dir_r4 = 'salida_reto_4'
os.makedirs(output_dir_r4, exist_ok=True)

# 1. Simular 30 valores enteros representando el número de vehículos por hora (entre 100 y 300)
np.random.seed(42) # Para reproducibilidad
num_dias_r4 = 30
vehiculos_por_hora_simulados_r4 = np.random.randint(100, 301, size=num_dias_r4) # 301 para incluir 300

print("\nValores simulados de vehículos por hora (primeros 10 días) (Reto 4):")
print(vehiculos_por_hora_simulados_r4[:10])

# 2. Crear una tabla de frecuencia agrupando por rangos
bins_r4 = [100, 150, 200, 250, 301] # Ajustar el último bin para incluir 300
labels_r4 = ['100-149', '150-199', '200-249', '250-300']
frecuencia_rangos_r4 = pd.cut(vehiculos_por_hora_simulados_r4, bins=bins_r4, labels=labels_r4, right=False, include_lowest=True).value_counts().sort_index()

print("\nReto 4: Tabla de Frecuencia por Rangos de Vehículos:")
print(frecuencia_rangos_r4)

# 3. Graficar los resultados usando un gráfico de barras
plt.figure(figsize=(10, 6))
frecuencia_rangos_r4.plot(kind='bar', color='lightgreen')
plt.title('Reto 4: Frecuencia de Conteo Vehicular por Rango Horario (7:00-8:00 am)')
plt.xlabel('Rango de Vehículos')
plt.ylabel('Frecuencia (Número de Días)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_r4, 'reto4_frecuencia_vehiculos.png')) # Guardar figura
plt.show()

# 4. Calcular la media y la moda del número de vehículos
media_vehiculos_r4 = np.mean(vehiculos_por_hora_simulados_r4)
moda_resultado_r4 = mode(vehiculos_por_hora_simulados_r4, keepdims=False)
moda_vehiculos_r4 = moda_resultado_r4.mode

print(f"\nReto 4: Media del número de vehículos: {media_vehiculos_r4:.2f}")
print(f"Reto 4: Moda del número de vehículos: {moda_vehiculos_r4}")

# 5. Analizar si hay evidencia de congestión recurrente
print("\nReto 4: Análisis de Congestión Recurrente:")
umbral_congestión_r4 = 250
dias_con_congestión_r4 = np.sum(vehiculos_por_hora_simulados_r4 >= umbral_congestión_r4)
proporcion_dias_con_congestión_r4 = dias_con_congestión_r4 / num_dias_r4

print(f"Número de días con {umbral_congestión_r4} o más vehículos: {dias_con_congestión_r4} de {num_dias_r4}")
print(f"Proporción de días con congestión: {proporcion_dias_con_congestión_r4:.2%}")

if proporcion_dias_con_congestión_r4 > 0.20:
    print("- Esto sugiere evidencia de **congestión recurrente** durante el horario de 7:00 a 8:00 am.")
    print("  Es un patrón que merece atención para la gestión del tráfico.")
else:
    print("- La evidencia de congestión recurrente no es fuerte en este conjunto de datos.")
    print("  Aunque el monitoreo continuo es importante, no parece haber un problema de saturación constante.")

# 6. Proponer una recomendación de mejora si hay saturación
print("\nReto 4: Recomendación de mejora (si hay saturación):")
if proporcion_dias_con_congestión_r4 > 0.20:
    print("- Dado que hay evidencia de congestión recurrente, se podría implementar una o varias de las siguientes medidas:")
    print("  - **Análisis de horarios de trabajo flexibles**: Promover que las empresas cercanas al puente ofrezcan")
    print("    horarios de entrada y salida escalonados para sus empleados, lo que podría distribuir el flujo vehicular.")
    print("  - **Mejora del transporte público**: Fortalecer la frecuencia y capacidad de las rutas de transporte público")
    print("    que cruzan el puente para incentivar su uso y reducir la dependencia del vehículo particular.")
    print("  - **Optimización de semáforos**: Si existen semáforos en las inmediaciones del puente, revisar y optimizar")
    print("    sus tiempos para mejorar el flujo vehicular y reducir los embotellamientos.")
    print("  - **Información en tiempo real**: Proporcionar información sobre el estado del tráfico para que los conductores")
    print("    puedan tomar rutas alternativas o ajustar sus horarios.")
else:
    print("- Con los datos actuales, no hay una saturación evidente que requiera medidas drásticas. Sin embargo,")
    print("  se podría considerar una planificación a futuro para evitar problemas a medida que aumente el parque vehicular")
    print("  o los patrones de desplazamiento cambien.")
print("--- FIN RETO 4 ---")

# --- RETO 5: Altura de Colados de Concreto 📏 ---
print("\n--- INICIO RETO 5: Altura de Colados de Concreto ---")

# Crear carpeta de salida para Reto 5
output_dir_r5 = 'salida_reto_5'
os.makedirs(output_dir_r5, exist_ok=True)

# Parámetros de la distribución de la altura de colados
media_altura_r5 = 1.5 # metros
sigma_altura_r5 = 0.2 # metros (desviación estándar)

# 1. Graficar la función de densidad normal con media 1.5 y sigma 0.2
x_pdf_r5 = np.linspace(media_altura_r5 - 3*sigma_altura_r5, media_altura_r5 + 3*sigma_altura_r5, 500)
pdf_teorica_r5 = norm.pdf(x_pdf_r5, loc=media_altura_r5, scale=sigma_altura_r5)

plt.figure(figsize=(10, 6))
plt.plot(x_pdf_r5, pdf_teorica_r5, color='purple', label=f'PDF Normal (μ={media_altura_r5} m, σ={sigma_altura_r5} m)')
plt.title('Reto 5: Función de Densidad de Probabilidad (PDF) para la Altura de Colados')
plt.xlabel('Altura (metros)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r5, 'reto5_pdf_altura.png')) # Guardar figura
plt.show()

# 2. Simular 200 mediciones de altura y representarlas como histograma con KDE
np.random.seed(42) # Para reproducibilidad
num_mediciones_r5 = 200
mediciones_altura_simuladas_r5 = np.random.normal(loc=media_altura_r5, scale=sigma_altura_r5, size=num_mediciones_r5)

print("\nPrimeros 10 mediciones de altura simuladas (Reto 5):")
print(mediciones_altura_simuladas_r5[:10].round(2))

plt.figure(figsize=(10, 6))
sns.histplot(mediciones_altura_simuladas_r5, kde=True, bins=15, stat='density', color='lightgreen', label='Mediciones Simuladas con KDE')
plt.title('Reto 5: Histograma de Mediciones de Altura con KDE')
plt.xlabel('Altura (metros)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r5, 'reto5_histograma_altura.png')) # Guardar figura
plt.show()

# 3. Superponer la curva normal teórica sobre los datos simulados
plt.figure(figsize=(10, 6))
sns.histplot(mediciones_altura_simuladas_r5, kde=True, bins=15, stat='density', color='lightgreen', alpha=0.7, label='Histograma Simulado')
plt.plot(x_pdf_r5, pdf_teorica_r5, color='purple', linestyle='--', label='PDF Normal Teórica')
plt.title('Reto 5: Histograma y PDF Teórica Superpuestos para Altura de Colados')
plt.xlabel('Altura (metros)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r5, 'reto5_histograma_pdf_superpuestos.png')) # Guardar figura
plt.show()

# 4. Calcular la probabilidad de que una medición esté entre 1.4 y 1.6 m
limite_inferior_r5 = 1.4 # metros
limite_superior_r5 = 1.6 # metros

prob_entre_1_4_y_1_6_r5 = norm.cdf(limite_superior_r5, loc=media_altura_r5, scale=sigma_altura_r5) - \
                       norm.cdf(limite_inferior_r5, loc=media_altura_r5, scale=sigma_altura_r5)
print(f"\nReto 5: Probabilidad de que una medición esté entre {limite_inferior_r5} y {limite_superior_r5} m: {prob_entre_1_4_y_1_6_r5:.4f}")

# 5. Sombrear el área correspondiente en la curva (con porcentaje en el título)
plt.figure(figsize=(10, 6))
plt.plot(x_pdf_r5, pdf_teorica_r5, color='purple', label=f'PDF Normal (μ={media_altura_r5} m, σ={sigma_altura_r5} m)')
x_sombreado_r5 = np.linspace(limite_inferior_r5, limite_superior_r5, 100)
plt.fill_between(x_sombreado_r5, 0, norm.pdf(x_sombreado_r5, loc=media_altura_r5, scale=sigma_altura_r5),
                 color='orange', alpha=0.5, label=f'P({limite_inferior_r5} < Altura < {limite_superior_r5} m)')
plt.axvline(x=limite_inferior_r5, color='blue', linestyle=':', label=f'{limite_inferior_r5} m')
plt.axvline(x=limite_superior_r5, color='blue', linestyle=':', label=f'{limite_superior_r5} m')

plt.title(f'Reto 5: Probabilidad de Altura entre {limite_inferior_r5} y {limite_superior_r5} m ({prob_entre_1_4_y_1_6_r5:.2%})')
plt.xlabel('Altura (metros)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r5, 'reto5_prob_altura_rango.png')) # Guardar figura
plt.show()

# --- Nuevo gráfico combinado para Reto 5 ---
plt.figure(figsize=(12, 7))
plt.plot(x_pdf_r5, pdf_teorica_r5, color='purple', linewidth=2, label=f'PDF Normal (μ={media_altura_r5} m, σ={sigma_altura_r5} m)')

# Línea de la media
plt.axvline(x=media_altura_r5, color='gray', linestyle='--', label=f'Media (μ = {media_altura_r5} m)')

# Líneas de los límites y área sombreada
plt.axvline(x=limite_inferior_r5, color='blue', linestyle=':', label=f'Límite Inferior ({limite_inferior_r5} m)')
plt.axvline(x=limite_superior_r5, color='blue', linestyle=':', label=f'Límite Superior ({limite_superior_r5} m)')
plt.fill_between(x_sombreado_r5, 0, norm.pdf(x_sombreado_r5, loc=media_altura_r5, scale=sigma_altura_r5),
                 color='orange', alpha=0.5, label=f'Área de Probabilidad ({prob_entre_1_4_y_1_6_r5:.2%})')

# Añadir el porcentaje de probabilidad en el gráfico
plt.text(media_altura_r5, norm.pdf(media_altura_r5, loc=media_altura_r5, scale=sigma_altura_r5) * 0.5,
         f'{prob_entre_1_4_y_1_6_r5:.2%}', horizontalalignment='center', color='darkgreen', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

plt.title('Reto 5: Análisis Completo de la Altura de Colados de Concreto')
plt.xlabel('Altura (metros)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r5, 'reto5_analisis_completo.png')) # Guardar figura
plt.show()

# 6. Evaluar si el proceso es lo suficientemente preciso
print("\nReto 5: Evaluación de la Precisión del Proceso:")
print(f"La probabilidad de que una medición esté entre {limite_inferior_r5} y {limite_superior_r5} m es del {prob_entre_1_4_y_1_6_r5:.2%}.")

if prob_entre_1_4_y_1_6_r5 > 0.60: # Umbral de ejemplo para "preciso"
    print("Esto indica que una gran proporción de los colados está dentro del rango deseado. El proceso tiene")
    print("una precisión aceptable para el objetivo de estar cerca de 1.5m, si la tolerancia de +/- 0.1m es adecuada.")
    print("La mayoría de los valores caen dentro de este rango, lo cual es positivo.")
else:
    print("Esto sugiere que el proceso podría no ser lo suficientemente preciso para mantener consistentemente")
    print("la altura de colado dentro de un rango tan estrecho (1.4m a 1.6m) alrededor de 1.5m.")
    print("Se deberían revisar los equipos de medición, la capacitación del personal o los procedimientos de colado")
    print("para mejorar la precisión y reducir la variabilidad (desviación estándar) y así aumentar la proporción")
    print("de colados dentro del rango objetivo.")

# --- Tabla de Resultados para Reto 5 ---
resultados_r5 = pd.DataFrame({
    'Métrica': ['Media (μ)', 'Desviación Estándar (σ)', 'Rango de Interés', 'Probabilidad Calculada', 'Evaluación de Precisión'],
    'Valor': [f'{media_altura_r5} m', f'{sigma_altura_r5} m', f'{limite_inferior_r5}m a {limite_superior_r5}m',
              f'{prob_entre_1_4_y_1_6_r5:.2%}',
              'No es suficientemente preciso para un rango estricto' if prob_entre_1_4_y_1_6_r5 <= 0.60 else 'Precisión aceptable']
})
print("\nReto 5: Tabla de Resultados Clave:")
print(resultados_r5.to_string(index=False)) # to_string para asegurar que se imprima todo el DataFrame
print("--- FIN RETO 5 ---")

# --- RETO 6: Tiempo de Fraguado de Cemento (Comparación de dos marcas) 🧪 ---
print("\n--- INICIO RETO 6: Tiempo de Fraguado de Cemento (Comparación) ---")

# Crear carpeta de salida para Reto 6
output_dir_r6 = 'salida_reto_6'
os.makedirs(output_dir_r6, exist_ok=True)

# Parámetros para Marca A
media_A_r6 = 45 # min
sigma_A_r6 = 5 # min

# Parámetros para Marca B
media_B_r6 = 50 # min
sigma_B_r6 = 6 # min

# 1. Simular 150 tiempos para Marca A y Marca B
np.random.seed(42) # Para reproducibilidad
num_tiempos_r6 = 150
tiempos_fraguado_A_r6 = np.random.normal(loc=media_A_r6, scale=sigma_A_r6, size=num_tiempos_r6)
tiempos_fraguado_B_r6 = np.random.normal(loc=media_B_r6, scale=sigma_B_r6, size=num_tiempos_r6)

print("\nPrimeros 10 tiempos de fraguado simulados para Marca A (Reto 6):")
print(tiempos_fraguado_A_r6[:10].round(2))
print("\nPrimeros 10 tiempos de fraguado simulados para Marca B (Reto 6):")
print(tiempos_fraguado_B_r6[:10].round(2))

# Rangos para graficar las PDFs (ajustar para cubrir ambas distribuciones)
min_val_r6 = min(media_A_r6 - 3*sigma_A_r6, media_B_r6 - 3*sigma_B_r6)
max_val_r6 = max(media_A_r6 + 3*sigma_A_r6, media_B_r6 + 3*sigma_B_r6)
x_pdf_general_r6 = np.linspace(min_val_r6, max_val_r6, 500)

# Calcular PDFs
pdf_A_r6 = norm.pdf(x_pdf_general_r6, loc=media_A_r6, scale=sigma_A_r6)
pdf_B_r6 = norm.pdf(x_pdf_general_r6, loc=media_B_r6, scale=sigma_B_r6)

# 2. Graficar ambas distribuciones normales en un solo gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_pdf_general_r6, pdf_A_r6, color='blue', label=f'Marca A (μ={media_A_r6} min, σ={sigma_A_r6} min)')
plt.plot(x_pdf_general_r6, pdf_B_r6, color='red', linestyle='--', label=f'Marca B (μ={media_B_r6} min, σ={sigma_B_r6} min)')
plt.title('Reto 6: Distribución de Tiempos de Fraguado para Marca A y Marca B')
plt.xlabel('Tiempo de Fraguado (minutos)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r6, 'reto6_distribucion_fraguado.png')) # Guardar figura
plt.show()

# 3. Calcular la probabilidad de que el fraguado sea menor a 47 min para cada marca
tiempo_critico_r6 = 47 # min
prob_A_menor_47_r6 = norm.cdf(tiempo_critico_r6, loc=media_A_r6, scale=sigma_A_r6)
prob_B_menor_47_r6 = norm.cdf(tiempo_critico_r6, loc=media_B_r6, scale=sigma_B_r6)

print(f"\nReto 6: Probabilidad de que el fraguado de Marca A sea menor a {tiempo_critico_r6} min: {prob_A_menor_47_r6:.4f}")
print(f"Reto 6: Probabilidad de que el fraguado de Marca B sea menor a {tiempo_critico_r6} min: {prob_B_menor_47_r6:.4f}")

# 4. Marcar esas probabilidades en el gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_pdf_general_r6, pdf_A_r6, color='blue', label=f'Marca A (μ={media_A_r6} min, σ={sigma_A_r6} min)')
plt.plot(x_pdf_general_r6, pdf_B_r6, color='red', linestyle='--', label=f'Marca B (μ={media_B_r6} min, σ={sigma_B_r6} min)')

x_somb_A_r6 = np.linspace(x_pdf_general_r6.min(), tiempo_critico_r6, 100)
plt.fill_between(x_somb_A_r6, 0, norm.pdf(x_somb_A_r6, loc=media_A_r6, scale=sigma_A_r6),
                 color='lightblue', alpha=0.5, label=f'P(Marca A < {tiempo_critico_r6} min)')
plt.axvline(x=tiempo_critico_r6, color='green', linestyle=':', label=f'{tiempo_critico_r6} minutos')
plt.text(tiempo_critico_r6, norm.pdf(tiempo_critico_r6, loc=media_A_r6, scale=sigma_A_r6) + 0.005,
         f'{prob_A_menor_47_r6:.2f}', color='blue', ha='right', va='bottom', fontsize=9)

x_somb_B_r6 = np.linspace(x_pdf_general_r6.min(), tiempo_critico_r6, 100)
plt.fill_between(x_somb_B_r6, 0, norm.pdf(x_somb_B_r6, loc=media_B_r6, scale=sigma_B_r6),
                 color='lightcoral', alpha=0.5, label=f'P(Marca B < {tiempo_critico_r6} min)')
plt.text(tiempo_critico_r6, norm.pdf(tiempo_critico_r6, loc=media_B_r6, scale=sigma_B_r6) + 0.005,
         f'{prob_B_menor_47_r6:.2f}', color='red', ha='left', va='bottom', fontsize=9)

plt.title(f'Reto 6: Distribución de Tiempos de Fraguado y Probabilidades (< {tiempo_critico_r6} min)')
plt.xlabel('Tiempo de Fraguado (minutos)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir_r6, 'reto6_prob_fraguado_47min.png')) # Guardar figura
plt.show()

# 5. Analizar cuál marca presenta menor variabilidad (σ) y mayor confiabilidad
print("\nReto 6: Análisis de Variabilidad y Confiabilidad:")
print(f"- Desviación estándar Marca A (σ_A): {sigma_A_r6} min")
print(f"- Desviación estándar Marca B (σ_B): {sigma_B_r6} min")

if sigma_A_r6 < sigma_B_r6:
    print(f"La Marca A ({sigma_A_r6} min) presenta **menor variabilidad** que la Marca B ({sigma_B_r6} min).")
    print("Una menor variabilidad implica una mayor **confiabilidad** y predictibilidad en el tiempo de fraguado.")
else:
    print(f"La Marca B ({sigma_B_r6} min) presenta **menor variabilidad** que la Marca A ({sigma_A_r6} min).")
    print("Una menor variabilidad implica una mayor **confiabilidad** y predictibilidad en el tiempo de fraguado.")

# 6. Concluir cuál se recomienda usar si se busca un fraguado más rápido y predecible
print("\nReto 6: Recomendación:")
print("- Para un fraguado **más rápido**, se busca una media de tiempo menor:")
print(f"  - Marca A: Media = {media_A_r6} min")
print(f"  - Marca B: Media = {media_B_r6} min")
print("  La Marca A es más rápida.")

print("- Para un fraguado **más predecible**, se busca menor variabilidad (menor σ):")
print(f"  - Marca A: Desviación Estándar = {sigma_A_r6} min")
print(f"  - Marca B: Desviación Estándar = {sigma_B_r6} min")
print("  La Marca A es más predecible.")

print("\nConclusión: Si se busca un fraguado **más rápido y predecible**, se recomienda usar la **Marca A**.")
print("--- FIN RETO 6 ---")

# --- Carpeta de salida para Reto 25 (solicitud específica del usuario) ---
# Aunque el contenido del Reto 25 no está definido en este código, se crea la carpeta si es solicitada.
output_dir_reto25 = 'salida_reto_25'
os.makedirs(output_dir_reto25, exist_ok=True)
print(f"\nSe ha creado la carpeta de salida '{output_dir_reto25}' según su solicitud.")
