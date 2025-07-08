# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os # Para manejar rutas de archivos

# Estilos de grafico (opcional, si quieres usar los estilos de seaborn)
# plt.style.use('seaborn-v0_8-darkgrid') # Usar la versión actualizada de seaborn style

# Asegúrate de que las carpetas 'input' y 'output' existan
os.makedirs('input', exist_ok=True)
os.makedirs('output', exist_ok=True)

def ejecutar_analisis_ejercicio1():
    """
    Ejercicio 1: Efecto de la cafeína en la velocidad de escritura
    Contexto: Empresa editorial quiere saber si tomar café aumenta la velocidad de escritura.
    Objetivo: Determinar si la cafeína tiene un efecto significativo.
    Datos simulados esperados: CSV con columnas: Participante, Grupo (Cafeína / SinCafeína), Velocidad (palabras/minuto)
    """
    print("\n--- Ejercicio 1: Efecto de la cafeína en la velocidad de escritura ---")
    # Problema: ¿La cafeína aumenta la velocidad de escritura (palabras por minuto)?
    # Hipótesis nula (H₀): Tomar cafeína no cambia la velocidad de escritura.
    # Hipótesis alternativa (H₁): Tomar cafeína aumenta la velocidad de escritura.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(42) # Para reproducibilidad
    n = 100 # Número de participantes por grupo

    # Simular puntuaciones de velocidad de escritura
    # Supongamos que la cafeína mejora la velocidad (media más alta)
    velocidad_cafeina = np.random.normal(loc=65, scale=8, size=n).round(2)
    velocidad_sin_cafeina = np.random.normal(loc=58, scale=8, size=n).round(2)

    # Crear un DataFrame
    datos_ejercicio1 = pd.DataFrame({
        'Participante': [f"P{i+1}" for i in range(n)] + [f"P{i+n+1}" for i in range(n)],
        "Grupo": ['Cafeína'] * n + ['SinCafeína'] * n,
        'Velocidad': np.concatenate([velocidad_cafeina, velocidad_sin_cafeina])
    })

    # Guardar los datos simulados en la carpeta 'input'
    datos_ejercicio1.to_csv('input/data_ejercicio1.csv', index=False)
    print("Datos simulados para Ejercicio 1 guardados en 'input/data_ejercicio1.csv'")
    print(datos_ejercicio1.head())

    # Paso 3: Define las variables
    # Variable independiente: Grupo (Cafeína, SinCafeína)
    # Variable dependiente: Velocidad (palabras/minuto)

    # Paso 4: Diseñar el experimento (ya implícito en la simulación)
    # Experimento: Asignar aleatoriamente a los participantes a dos grupos y medir su velocidad de escritura.

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio1 = datos_ejercicio1.groupby('Grupo')['Velocidad'].describe()
    print("\nResumen descriptivo de la Velocidad por Grupo:")
    print(resumen_ejercicio1)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio1.boxplot(column='Velocidad', by='Grupo', grid=False, patch_artist=True)
    plt.title('Velocidad de Escritura por Grupo (Cafeína vs. Sin Cafeína)')
    plt.suptitle('') # Elimina el título superior que a veces genera boxplot
    plt.xlabel('Grupo')
    plt.ylabel('Velocidad (palabras/minuto)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio1_velocidad_cafeina.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05 # Nivel de significancia

    grupo_cafeina = datos_ejercicio1[datos_ejercicio1['Grupo'] == 'Cafeína']['Velocidad']
    grupo_sin_cafeina = datos_ejercicio1[datos_ejercicio1['Grupo'] == 'SinCafeína']['Velocidad']

    # Realizar la prueba t de Student para muestras independientes
    # equal_var=False es para la prueba t de Welch, que es más robusta si las varianzas no son iguales.
    t_stat, p_value = stats.ttest_ind(grupo_cafeina, grupo_sin_cafeina, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en la velocidad de escritura entre el grupo con cafeína y el grupo sin cafeína (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que la cafeína tiene un efecto estadísticamente significativo en la velocidad de escritura (p >= {alpha}).")
    print("-" * 60)


def ejecutar_analisis_ejercicio2():
    """
    Ejercicio 2: Entrenamiento físico y memoria
    Contexto: Firma de bienestar corporativo quiere saber si una pausa de ejercicio mejora la memoria.
    Objetivo: Diseñar y analizar un experimento para probar si el ejercicio tiene efecto significativo.
    Datos simulados esperados: CSV con columnas: Empleado, Condición (Ejercicio / Reposo), PuntajeMemoria (0-100)
    """
    print("\n--- Ejercicio 2: Entrenamiento físico y memoria ---")
    # Problema: ¿Una pausa de ejercicio físico ligero mejora la memoria a corto plazo?
    # Hipótesis nula (H₀): El ejercicio físico ligero no cambia las puntuaciones de memoria a corto plazo.
    # Hipótesis alternativa (H₁): El ejercicio físico ligero mejora las puntuaciones de memoria a corto plazo.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(43) # Para reproducibilidad
    n = 80 # Número de empleados por grupo

    # Simular puntuaciones de memoria
    # Suponemos que el ejercicio mejora la memoria (media más alta)
    puntaje_ejercicio = np.random.normal(loc=80, scale=12, size=n).round(2)
    puntaje_reposo = np.random.normal(loc=72, scale=12, size=n).round(2)

    # Crear un DataFrame
    datos_ejercicio2 = pd.DataFrame({
        'Empleado': [f"E{i+1}" for i in range(n)] + [f"E{i+n+1}" for i in range(n)],
        "Condición": ['Ejercicio'] * n + ['Reposo'] * n,
        'PuntajeMemoria': np.concatenate([puntaje_ejercicio, puntaje_reposo])
    })

    # Guardar los datos simulados
    datos_ejercicio2.to_csv('input/data_ejercicio2.csv', index=False)
    print("Datos simulados para Ejercicio 2 guardados en 'input/data_ejercicio2.csv'")
    print(datos_ejercicio2.head())

    # Paso 3: Define las variables
    # Variable independiente: Condición (Ejercicio, Reposo)
    # Variable dependiente: PuntajeMemoria

    # Paso 4: Diseñar el experimento (implícito)

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio2 = datos_ejercicio2.groupby('Condición')['PuntajeMemoria'].describe()
    print("\nResumen descriptivo del Puntaje de Memoria por Condición:")
    print(resumen_ejercicio2)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio2.boxplot(column='PuntajeMemoria', by='Condición', grid=False, patch_artist=True)
    plt.title('Puntajes de Memoria por Condición (Ejercicio vs. Reposo)')
    plt.suptitle('')
    plt.xlabel('Condición')
    plt.ylabel('Puntaje de Memoria (0-100)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio2_memoria_ejercicio.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05

    grupo_ejercicio = datos_ejercicio2[datos_ejercicio2['Condición'] == 'Ejercicio']['PuntajeMemoria']
    grupo_reposo = datos_ejercicio2[datos_ejercicio2['Condición'] == 'Reposo']['PuntajeMemoria']

    t_stat, p_value = stats.ttest_ind(grupo_ejercicio, grupo_reposo, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en el puntaje de memoria entre el grupo con ejercicio y el grupo en reposo (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que el ejercicio físico tiene un efecto estadísticamente significativo en la memoria a corto plazo (p >= {alpha}).")
    print("-" * 60)

def ejecutar_analisis_ejercicio3():
    """
    Ejercicio 3: Música instrumental vs. concentración
    Contexto: Una app de productividad solicita un estudio sobre si la música instrumental de fondo mejora la concentración en tareas lógicas.
    Objetivo: Analizar si existe una diferencia significativa en el tiempo de resolución de un test lógico con y sin música.
    Datos simulados esperados: CSV con columnas: Usuario, Grupo (Música / Silencio), TiempoTest (segundos)
    """
    print("\n--- Ejercicio 3: Música instrumental vs. concentración ---")
    # Problema: ¿La música instrumental de fondo mejora la concentración en tareas lógicas (reduce el tiempo de resolución)?
    # Hipótesis nula (H₀): Escuchar música instrumental no cambia el tiempo de resolución de un test lógico.
    # Hipótesis alternativa (H₁): Escuchar música instrumental reduce el tiempo de resolución de un test lógico.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(44) # Para reproducibilidad
    n = 120 # Número de usuarios por grupo

    # Simular tiempos de resolución (menor tiempo es mejor)
    # Suponemos que la música reduce el tiempo de resolución (media más baja)
    tiempo_musica = np.random.normal(loc=150, scale=30, size=n).round(2) # Segundos
    tiempo_silencio = np.random.normal(loc=170, scale=30, size=n).round(2) # Segundos

    # Crear un DataFrame
    datos_ejercicio3 = pd.DataFrame({
        'Usuario': [f"U{i+1}" for i in range(n)] + [f"U{i+n+1}" for i in range(n)],
        "Grupo": ['Música'] * n + ['Silencio'] * n,
        'TiempoTest': np.concatenate([tiempo_musica, tiempo_silencio])
    })

    # Guardar los datos simulados
    datos_ejercicio3.to_csv('input/data_ejercicio3.csv', index=False)
    print("Datos simulados para Ejercicio 3 guardados en 'input/data_ejercicio3.csv'")
    print(datos_ejercicio3.head())

    # Paso 3: Define las variables
    # Variable independiente: Grupo (Música, Silencio)
    # Variable dependiente: TiempoTest (segundos)

    # Paso 4: Diseñar el experimento (implícito)

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio3 = datos_ejercicio3.groupby('Grupo')['TiempoTest'].describe()
    print("\nResumen descriptivo del Tiempo de Test por Grupo:")
    print(resumen_ejercicio3)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio3.boxplot(column='TiempoTest', by='Grupo', grid=False, patch_artist=True)
    plt.title('Tiempo de Resolución de Test Lógico por Grupo (Música vs. Silencio)')
    plt.suptitle('')
    plt.xlabel('Grupo')
    plt.ylabel('Tiempo de Test (segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio3_musica_concentracion.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05

    grupo_musica = datos_ejercicio3[datos_ejercicio3['Grupo'] == 'Música']['TiempoTest']
    grupo_silencio = datos_ejercicio3[datos_ejercicio3['Grupo'] == 'Silencio']['TiempoTest']

    t_stat, p_value = stats.ttest_ind(grupo_musica, grupo_silencio, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en el tiempo de resolución del test lógico entre el grupo con música y el grupo en silencio (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que la música instrumental tiene un efecto estadísticamente significativo en el tiempo de resolución del test lógico (p >= {alpha}).")
    print("-" * 60)

def ejecutar_analisis_ejercicio4():
    """
    Ejercicio 4: Efecto del diseño de una app en la rapidez de compra
    Contexto: Empresa de e-commerce prueba dos versiones de su app (A y B).
    Objetivo: Comparar los tiempos medios entre ambas versiones usando inferencia estadística.
    Datos simulados esperados: CSV con columnas: cliente, VersiónApp (A/B), TiempoCompra (segundos)
    """
    print("\n--- Ejercicio 4: Efecto del diseño de una app en la rapidez de compra ---")
    # Problema: ¿Una versión de la app (A o B) permite completar la compra más rápido?
    # Hipótesis nula (H₀): El diseño de la app no afecta la rapidez de compra (los tiempos medios son iguales).
    # Hipótesis alternativa (H₁): Existe una diferencia significativa en la rapidez de compra entre las versiones A y B de la app.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(45) # Para reproducibilidad
    n = 150 # Número de clientes por versión

    # Simular tiempos de compra (menor tiempo es mejor)
    # Suponemos que la Versión A es ligeramente más rápida
    tiempo_version_a = np.random.normal(loc=70, scale=15, size=n).round(2) # Segundos
    tiempo_version_b = np.random.normal(loc=75, scale=15, size=n).round(2) # Segundos

    # Crear un DataFrame
    datos_ejercicio4 = pd.DataFrame({
        'Cliente': [f"C{i+1}" for i in range(n)] + [f"C{i+n+1}" for i in range(n)],
        "VersiónApp": ['A'] * n + ['B'] * n,
        'TiempoCompra': np.concatenate([tiempo_version_a, tiempo_version_b])
    })

    # Guardar los datos simulados
    datos_ejercicio4.to_csv('input/data_ejercicio4.csv', index=False)
    print("Datos simulados para Ejercicio 4 guardados en 'input/data_ejercicio4.csv'")
    print(datos_ejercicio4.head())

    # Paso 3: Define las variables
    # Variable independiente: VersiónApp (A, B)
    # Variable dependiente: TiempoCompra (segundos)

    # Paso 4: Diseñar el experimento (implícito)

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio4 = datos_ejercicio4.groupby('VersiónApp')['TiempoCompra'].describe()
    print("\nResumen descriptivo del Tiempo de Compra por Versión de App:")
    print(resumen_ejercicio4)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio4.boxplot(column='TiempoCompra', by='VersiónApp', grid=False, patch_artist=True)
    plt.title('Tiempo de Compra por Versión de App (A vs. B)')
    plt.suptitle('')
    plt.xlabel('Versión de App')
    plt.ylabel('Tiempo de Compra (segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio4_tiempo_compra_app.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05

    version_a = datos_ejercicio4[datos_ejercicio4['VersiónApp'] == 'A']['TiempoCompra']
    version_b = datos_ejercicio4[datos_ejercicio4['VersiónApp'] == 'B']['TiempoCompra']

    t_stat, p_value = stats.ttest_ind(version_a, version_b, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en el tiempo de compra entre la Versión A y la Versión B de la app (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que el diseño de una app afecta la rapidez de compra (p >= {alpha}).")
    print("-" * 60)

def ejecutar_analisis_ejercicio5():
    """
    Ejercicio 5: Refrigerio saludable y rendimiento académico
    Contexto: Una escuela evalúa si ofrecer una colación saludable antes del examen mejora el rendimiento.
    Objetivo: Evaluar si hay una diferencia significativa entre los grupos (con colación vs sin colación).
    Datos simulados esperados: CSV con columnas: Alumno, Grupo (Colación / SinColación), Calificación (0-100)
    """
    print("\n--- Ejercicio 5: Refrigerio saludable y rendimiento académico ---")
    # Problema: ¿Ofrecer una colación saludable antes del examen mejora el rendimiento académico?
    # Hipótesis nula (H₀): Ofrecer una colación saludable no cambia las calificaciones de los alumnos.
    # Hipótesis alternativa (H₁): Ofrecer una colación saludable mejora las calificaciones de los alumnos.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(46) # Para reproducibilidad
    n = 100 # Número de alumnos por grupo

    # Simular calificaciones (más alta es mejor)
    # Suponemos que la colación mejora las calificaciones
    calificacion_colacion = np.random.normal(loc=78, scale=10, size=n).round(2)
    calificacion_sin_colacion = np.random.normal(loc=70, scale=10, size=n).round(2)

    # Crear un DataFrame
    datos_ejercicio5 = pd.DataFrame({
        'Alumno': [f"A{i+1}" for i in range(n)] + [f"A{i+n+1}" for i in range(n)],
        "Grupo": ['Colación'] * n + ['SinColación'] * n,
        'Calificación': np.concatenate([calificacion_colacion, calificacion_sin_colacion])
    })

    # Guardar los datos simulados
    datos_ejercicio5.to_csv('input/data_ejercicio5.csv', index=False)
    print("Datos simulados para Ejercicio 5 guardados en 'input/data_ejercicio5.csv'")
    print(datos_ejercicio5.head())

    # Paso 3: Define las variables
    # Variable independiente: Grupo (Colación, SinColación)
    # Variable dependiente: Calificación (0-100)

    # Paso 4: Diseñar el experimento (implícito)

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio5 = datos_ejercicio5.groupby('Grupo')['Calificación'].describe()
    print("\nResumen descriptivo de la Calificación por Grupo:")
    print(resumen_ejercicio5)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio5.boxplot(column='Calificación', by='Grupo', grid=False, patch_artist=True)
    plt.title('Calificaciones por Grupo (Colación vs. Sin Colación)')
    plt.suptitle('')
    plt.xlabel('Grupo')
    plt.ylabel('Calificación (0-100)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio5_rendimiento_academico.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05

    grupo_colacion = datos_ejercicio5[datos_ejercicio5['Grupo'] == 'Colación']['Calificación']
    grupo_sin_colacion = datos_ejercicio5[datos_ejercicio5['Grupo'] == 'SinColación']['Calificación']

    t_stat, p_value = stats.ttest_ind(grupo_colacion, grupo_sin_colacion, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en las calificaciones entre el grupo con colación y el grupo sin colación (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que ofrecer una colación saludable mejora estadísticamente el rendimiento académico (p >= {alpha}).")
    print("-" * 60)

def ejecutar_analisis_ejercicio6():
    """
    Ejercicio 6: Cursos de comunicación y desempeño en ventas
    Contexto: Una empresa capacita a su personal en comunicación efectiva. Se desea comprobar si los vendedores capacitados logran más ventas.
    Objetivo: Comparar el promedio de ventas mensuales entre empleados capacitados y no capacitados.
    Datos simulados esperados: CSV con columnas: Empleado, Formación (Si/NO), VentasMensuales
    """
    print("\n--- Ejercicio 6: Cursos de comunicación y desempeño en ventas ---")
    # Problema: ¿Los cursos de comunicación efectiva aumentan las ventas mensuales de los empleados?
    # Hipótesis nula (H₀): La formación en comunicación no cambia el promedio de ventas mensuales.
    # Hipótesis alternativa (H₁): La formación en comunicación aumenta el promedio de ventas mensuales.

    # 2. Crear el conjunto de datos simulados
    np.random.seed(47) # Para reproducibilidad
    n = 70 # Número de empleados por grupo

    # Simular ventas mensuales (más ventas es mejor)
    # Suponemos que la capacitación mejora las ventas
    ventas_si_formacion = np.random.normal(loc=120, scale=25, size=n).round(2)
    ventas_no_formacion = np.random.normal(loc=100, scale=25, size=n).round(2)

    # Crear un DataFrame
    datos_ejercicio6 = pd.DataFrame({
        'Empleado': [f"E{i+1}" for i in range(n)] + [f"E{i+n+1}" for i in range(n)],
        "Formación": ['Si'] * n + ['NO'] * n,
        'VentasMensuales': np.concatenate([ventas_si_formacion, ventas_no_formacion])
    })

    # Guardar los datos simulados
    datos_ejercicio6.to_csv('input/data_ejercicio6.csv', index=False)
    print("Datos simulados para Ejercicio 6 guardados en 'input/data_ejercicio6.csv'")
    print(datos_ejercicio6.head())

    # Paso 3: Define las variables
    # Variable independiente: Formación (Si, NO)
    # Variable dependiente: VentasMensuales

    # Paso 4: Diseñar el experimento (implícito)

    # Paso 5: Recolectar los datos (ya generados)
    # Paso 6: Analizar los datos
    resumen_ejercicio6 = datos_ejercicio6.groupby('Formación')['VentasMensuales'].describe()
    print("\nResumen descriptivo de las Ventas Mensuales por Grupo de Formación:")
    print(resumen_ejercicio6)

    # Paso 7: Visualizar los datos
    plt.figure(figsize=(8, 6))
    datos_ejercicio6.boxplot(column='VentasMensuales', by='Formación', grid=False, patch_artist=True)
    plt.title('Ventas Mensuales por Formación (Sí vs. No)')
    plt.suptitle('')
    plt.xlabel('Formación')
    plt.ylabel('Ventas Mensuales')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/boxplot_ejercicio6_ventas_formacion.png')
    plt.show()

    # Paso 8: Interpretar los resultados (Prueba T de Student)
    alpha = 0.05

    grupo_si_formacion = datos_ejercicio6[datos_ejercicio6['Formación'] == 'Si']['VentasMensuales']
    grupo_no_formacion = datos_ejercicio6[datos_ejercicio6['Formación'] == 'NO']['VentasMensuales']

    t_stat, p_value = stats.ttest_ind(grupo_si_formacion, grupo_no_formacion, equal_var=False)

    print(f"\nEstadístico T: {t_stat:.4f}")
    print(f"Valor P: {p_value:.4f}")

    if p_value < alpha:
        print("Conclusión: Rechazamos la hipótesis nula.")
        print(f"Existe una diferencia estadísticamente significativa en las ventas mensuales entre los empleados capacitados y no capacitados (p < {alpha}).")
    else:
        print("Conclusión: No rechazamos la hipótesis nula.")
        print(f"No hay evidencia suficiente para afirmar que los cursos de comunicación tienen un efecto estadísticamente significativo en el desempeño de ventas (p >= {alpha}).")
    print("-" * 60)

# Bloque principal para ejecutar todos los análisis
if __name__ == "__main__":
    ejecutar_analisis_ejercicio1()
    ejecutar_analisis_ejercicio2()
    ejecutar_analisis_ejercicio3()
    ejecutar_analisis_ejercicio4()
    ejecutar_analisis_ejercicio5()
    ejecutar_analisis_ejercicio6()