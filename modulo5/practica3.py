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

def generar_datos_y_analizar(ejercicio_num, contexto, objetivo, hipotesis_nula, hipotesis_alternativa,
                              columnas_csv, nombre_variable_ind, nombre_variable_dep,
                              nombre_grupo1, nombre_grupo2,
                              loc1, loc2, scale, n_participantes):
    """
    Función genérica para simular datos, realizar análisis y devolver los resultados formateados.
    """
    np.random.seed(40 + ejercicio_num) # Semilla diferente para cada ejercicio

    # Simular datos
    grupo1_data = np.random.normal(loc=loc1, scale=scale, size=n_participantes).round(2)
    grupo2_data = np.random.normal(loc=loc2, scale=scale, size=n_participantes).round(2)

    datos = pd.DataFrame({
        columnas_csv[0]: [f"P{i+1}" for i in range(n_participantes)] + [f"P{i+n_participantes+1}" for i in range(n_participantes)],
        nombre_variable_ind: [nombre_grupo1] * n_participantes + [nombre_grupo2] * n_participantes,
        nombre_variable_dep: np.concatenate([grupo1_data, grupo2_data])
    })

    # Guardar los datos simulados
    nombre_csv = f'input/data_ejercicio{ejercicio_num}.csv'
    datos.to_csv(nombre_csv, index=False)
    # print(f"Datos simulados para Ejercicio {ejercicio_num} guardados en '{nombre_csv}'")

    # Analizar datos
    resumen = datos.groupby(nombre_variable_ind)[nombre_variable_dep].describe()

    # Visualizar datos
    plt.figure(figsize=(8, 6))
    datos.boxplot(column=nombre_variable_dep, by=nombre_variable_ind, grid=False, patch_artist=True)
    plt.title(f'{nombre_variable_dep} por {nombre_variable_ind}')
    plt.suptitle('')
    plt.xlabel(nombre_variable_ind)
    plt.ylabel(nombre_variable_dep)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    nombre_boxplot = f'output/boxplot_ejercicio{ejercicio_num}.png'
    plt.savefig(nombre_boxplot)
    plt.close() # Cierra la figura para evitar que se muestren muchas ventanas al ejecutar

    # Interpretar resultados (Prueba T de Student)
    alpha = 0.05
    grupo1 = datos[datos[nombre_variable_ind] == nombre_grupo1][nombre_variable_dep]
    grupo2 = datos[datos[nombre_variable_ind] == nombre_grupo2][nombre_variable_dep]
    t_stat, p_value = stats.ttest_ind(grupo1, grupo2, equal_var=False) # Prueba t de Welch

    # Formatear resultados para el informe
    reporte = f"### Ejercicio {ejercicio_num}: {objetivo}\n\n"
    reporte += f"**Contexto:** {contexto}\n\n"
    reporte += f"**Pregunta de Investigación:** ¿{objetivo}?\n\n"
    reporte += f"**Hipótesis Nula (H₀):** {hipotesis_nula}\n\n"
    reporte += f"**Hipótesis Alternativa (H₁):** {hipotesis_alternativa}\n\n"
    reporte += f"**Variables:**\n"
    reporte += f"- **Variable Independiente:** {nombre_variable_ind} ({nombre_grupo1}, {nombre_grupo2})\n"
    reporte += f"- **Variable Dependiente:** {nombre_variable_dep}\n\n"
    reporte += f"**Resumen Descriptivo de los Datos ({nombre_csv}):**\n"
    reporte += f"```\n{resumen.to_string()}\n```\n\n"
    reporte += f"**Análisis Estadístico (Prueba t de Student para Muestras Independientes):**\n"
    reporte += f"- **Estadístico T:** {t_stat:.4f}\n"
    reporte += f"- **Valor P:** {p_value:.4f}\n"
    reporte += f"- **Nivel de Significación (α):** {alpha}\n\n"

    conclusion = ""
    if p_value < alpha:
        conclusion = (f"**Conclusión:** Se **rechaza la hipótesis nula** (p < {alpha}). "
                      f"Existe una diferencia estadísticamente significativa en '{nombre_variable_dep}' "
                      f"entre el grupo '{nombre_grupo1}' y el grupo '{nombre_grupo2}'. "
                      f"Esto sugiere que {objetivo}."
                      f"\n\n*Visualización en: {nombre_boxplot}*")
    else:
        conclusion = (f"**Conclusión:** No se **rechaza la hipótesis nula** (p >= {alpha}). "
                      f"No hay evidencia estadísticamente suficiente para afirmar una diferencia significativa en '{nombre_variable_dep}' "
                      f"entre el grupo '{nombre_grupo1}' y el grupo '{nombre_grupo2}'. "
                      f"Esto sugiere que {hipotesis_nula}."
                      f"\n\n*Visualización en: {nombre_boxplot}*")
    reporte += conclusion + "\n\n---\n"
    return reporte

def main_analisis_documento():
    """
    Función principal que ejecuta todos los análisis y compila el informe.
    """
    informe_completo = []

    informe_completo.append("# Informe de Análisis Estadístico de Datos Simulados\n\n")
    informe_completo.append("Este documento presenta el análisis estadístico de seis ejercicios diferentes, cada uno simulando un escenario de investigación y aplicando pruebas de hipótesis. El objetivo es determinar la existencia de efectos significativos entre grupos o condiciones utilizando el método científico y herramientas estadísticas en Python.\n\n")
    informe_completo.append("Cada sección detalla el contexto del problema, las hipótesis planteadas, las variables involucradas, un resumen descriptivo de los datos simulados, los resultados de la prueba t de Student y una conclusión basada en el valor p.\n\n")
    informe_completo.append("--- \n\n")

    # Ejercicio 1: Efecto de la cafeína en la velocidad de escritura
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=1,
        contexto="Empresa editorial. Quieren saber si tomar café antes de trabajar aumenta la velocidad de escritura.",
        objetivo="la cafeína tiene un efecto significativo en la velocidad de escritura",
        hipotesis_nula="Tomar cafeína no cambia la velocidad de escritura.",
        hipotesis_alternativa="Tomar cafeína aumenta la velocidad de escritura.",
        columnas_csv=['Participante', 'Grupo', 'Velocidad'],
        nombre_variable_ind='Grupo',
        nombre_variable_dep='Velocidad (palabras/minuto)',
        nombre_grupo1='Cafeína',
        nombre_grupo2='SinCafeína',
        loc1=65, loc2=58, scale=8, n_participantes=100
    ))

    # Ejercicio 2: Entrenamiento físico y memoria
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=2,
        contexto="Firma de bienestar corporativo. Un cliente quiere saber si una pausa de ejercicio físico ligero mejora la memoria a corto plazo en sus empleados.",
        objetivo="el ejercicio físico ligero mejora la memoria a corto plazo en sus empleados",
        hipotesis_nula="El ejercicio físico ligero no cambia las puntuaciones de memoria a corto plazo.",
        hipotesis_alternativa="El ejercicio físico ligero mejora las puntuaciones de memoria a corto plazo.",
        columnas_csv=['Empleado', 'Condición', 'PuntajeMemoria'],
        nombre_variable_ind='Condición',
        nombre_variable_dep='PuntajeMemoria (0-100)',
        nombre_grupo1='Ejercicio',
        nombre_grupo2='Reposo',
        loc1=80, loc2=72, scale=12, n_participantes=80
    ))

    # Ejercicio 3: Música instrumental vs. concentración
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=3,
        contexto="Una app de productividad solicita un estudio sobre si la música instrumental de fondo mejora la concentración en tareas lógicas.",
        objetivo="la música instrumental de fondo mejora la concentración en tareas lógicas (reduce el tiempo de resolución)",
        hipotesis_nula="Escuchar música instrumental no cambia el tiempo de resolución de un test lógico.",
        hipotesis_alternativa="Escuchar música instrumental reduce el tiempo de resolución de un test lógico.",
        columnas_csv=['Usuario', 'Grupo', 'TiempoTest'],
        nombre_variable_ind='Grupo',
        nombre_variable_dep='TiempoTest (segundos)',
        nombre_grupo1='Música',
        nombre_grupo2='Silencio',
        loc1=150, loc2=170, scale=30, n_participantes=120
    ))

    # Ejercicio 4: Efecto del diseño de una app en la rapidez de compra
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=4,
        contexto="Empresa de e-commerce prueba dos versiones de su app (A y B). Desean saber si una versión permite completar la compra más rápido.",
        objetivo="una versión de la app (A o B) permite completar la compra más rápido",
        hipotesis_nula="El diseño de la app no afecta la rapidez de compra (los tiempos medios son iguales).",
        hipotesis_alternativa="Existe una diferencia significativa en la rapidez de compra entre las versiones A y B de la app.",
        columnas_csv=['Cliente', 'VersiónApp', 'TiempoCompra'],
        nombre_variable_ind='VersiónApp',
        nombre_variable_dep='TiempoCompra (segundos)',
        nombre_grupo1='A',
        nombre_grupo2='B',
        loc1=70, loc2=75, scale=15, n_participantes=150
    ))

    # Ejercicio 5: Refrigerio saludable y rendimiento académico
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=5,
        contexto="Una escuela evalúa si ofrecer una colación saludable antes del examen mejora el rendimiento de sus alumnos.",
        objetivo="ofrecer una colación saludable antes del examen mejora el rendimiento académico",
        hipotesis_nula="Ofrecer una colación saludable no cambia las calificaciones de los alumnos.",
        hipotesis_alternativa="Ofrecer una colación saludable mejora las calificaciones de los alumnos.",
        columnas_csv=['Alumno', 'Grupo', 'Calificación'],
        nombre_variable_ind='Grupo',
        nombre_variable_dep='Calificación (0-100)',
        nombre_grupo1='Colación',
        nombre_grupo2='SinColación',
        loc1=78, loc2=70, scale=10, n_participantes=100
    ))

    # Ejercicio 6: Cursos de comunicación y desempeño en ventas
    informe_completo.append(generar_datos_y_analizar(
        ejercicio_num=6,
        contexto="Una empresa capacita a su personal en comunicación efectiva. Se desea comprobar si los vendedores capacitados logran más ventas.",
        objetivo="los cursos de comunicación efectiva aumentan las ventas mensuales de los empleados",
        hipotesis_nula="La formación en comunicación no cambia el promedio de ventas mensuales.",
        hipotesis_alternativa="La formación en comunicación aumenta el promedio de ventas mensuales.",
        columnas_csv=['Empleado', 'Formación', 'VentasMensuales'],
        nombre_variable_ind='Formación',
        nombre_variable_dep='VentasMensuales',
        nombre_grupo1='Si',
        nombre_grupo2='NO',
        loc1=120, loc2=100, scale=25, n_participantes=70
    ))

    # Conclusión General
    informe_completo.append("\n## Conclusión General\n\n")
    informe_completo.append("Este análisis ha permitido aplicar principios de estadística inferencial a diversos escenarios simulados. A través de la formulación de hipótesis, la simulación de datos, el cálculo de estadísticas descriptivas y la realización de pruebas t de Student, se han extraído conclusiones sobre la significancia de los efectos observados. Los resultados, junto con las visualizaciones generadas, ofrecen una base sólida para la toma de decisiones en cada uno de los contextos presentados. Es crucial recordar que, aunque los datos son simulados, la metodología empleada es replicable en situaciones con datos reales.\n")

    # Guardar el informe completo en un archivo de texto
    nombre_archivo_informe = 'output/informe_analisis_estadistico.txt'
    with open(nombre_archivo_informe, 'w', encoding='utf-8') as f:
        f.write("".join(informe_completo))
    print(f"\n¡Análisis completado! El informe detallado se ha guardado en: '{nombre_archivo_informe}'")
    print("También se han generado archivos CSV con los datos simulados en 'input/' y gráficos boxplot en 'output/'.")

# ¡CORRECCIÓN AQUÍ! Cambiado de "__main_" a "__main__"
if __name__ == "__main__":
    main_analisis_documento() # Llama a la función principal