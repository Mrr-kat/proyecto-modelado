# Proyecto 2: Propagación de una epidemia con Simulación Basada en Agentes

**Asignatura:** Modelado y Simulación
**Tema:** COVID-19 en Colombia — Modelo SEIR con red de contactos

## Descripción

Modelo de simulación basada en agentes (ABM) que reproduce la propagación del COVID-19 durante la primera ola en Colombia (2020-2021). Se calibra con datos reales de Our World in Data y se evalúan 3 escenarios de intervención.

## Estructura del proyecto

proyecto_epidemia/
├── data/
│ ├── raw/covid_owid.csv # Dataset original
│ └── processed/covid_limpio.csv # Datos filtrados (Colombia)
├── src/
│ ├── descargar_datos.py # Descarga el dataset
│ ├── explorar.py # Exploración inicial
│ ├── analisis_datos.py # Estadísticas y gráficos
│ ├── modelo_epidemia.py # Modelo ABM (Mesa)
│ ├── calibracion.py # Ajuste de β
│ ├── validacion.py # Validación vs. datos reales
│ ├── escenarios.py # 3 escenarios + base
│ ├── emergente.py # Análisis de comportamiento emergente
│ └── generar_diapositivas.py # Genera el .pptx automáticamente
├── results/
│ ├── figuras/ # Gráficos PNG
│ └── tablas/ # Tablas CSV y reportes TXT
├── presentacion/
│ └── diapositivas.pptx # ENTREGABLE FINAL
└── requirements.txt

Ejecución
# 1. Descargar dataset
python src/descargar_datos.py

# 2. Análisis exploratorio
python src/analisis_datos.py

# 3. Calibrar β
python -m src.calibracion

# 4. Validar
python -m src.validacion

# 5. Ejecutar escenarios
python -m src.escenarios

# 6. Analizar comportamiento emergente
python -m src.emergente

# 7. Generar diapositivas
python -m src.generar_diapositivas

Resultados principales
Escenario	Pico	Reducción vs base
Base	462	—
Esc1 (vacunación 30%)	142	69.3%
Esc2 (cuarentena 1%)	54	88.3%
Esc3 (vacunación + cuarentena)	14	96.9%
Modelo
Tipo: SEIR sobre red small-world (Watts-Strogatz)

Agentes: 1000 personas con estados S, E, I, R, V

Parámetros calibrados: β = 0.045, incubación = 3 días, infeccioso = 7 días

R₀ efectivo: ≈ 2.8 (consistente con COVID-19 real)

Validación
Correlación modelo vs. datos reales: 0.41

MSE: 0.115 | MAE: 0.231

El modelo reproduce la dinámica cualitativa (curva S)

Referencias
Our World in Data COVID-19: https://github.com/owid/covid-19-data

Mesa: https://mesa.readthedocs.io

NetworkX: https://networkx.org