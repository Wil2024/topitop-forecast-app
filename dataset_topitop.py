import numpy as np
import pandas as pd

# Configurar semilla para reproducibilidad
np.random.seed(42)

# 1. Generar fechas mensuales (2020-2024)
dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="ME")  # 'ME' = Month End
n_periods = len(dates)

# 2. Componentes de la serie temporal (en Soles)
PRECIO_ABRIGO = 79  # Precio unitario fijo
trend = np.linspace(100 * PRECIO_ABRIGO, 100 * PRECIO_ABRIGO * (1.08)**5, n_periods)  # Tendencias en Soles

# Estacionalidad: Picos en Diciembre y Julio (en Soles)
seasonality = np.zeros(n_periods)
for i, date in enumerate(dates):
    if date.month == 12:  # Navidad
        seasonality[i] = 50 * PRECIO_ABRIGO  # +50 abrigos = +50*79 S/
    elif date.month == 7:  # Liquidación invierno
        seasonality[i] = 30 * PRECIO_ABRIGO

# Eventos especiales (en Soles)
events = np.zeros(n_periods)
for i, date in enumerate(dates):
    if date.month == 5:  # Hot Sale
        events[i] = np.random.normal(20 * PRECIO_ABRIGO, 5 * PRECIO_ABRIGO)
    elif date.month == 11:  # Black Friday
        events[i] = np.random.normal(40 * PRECIO_ABRIGO, 10 * PRECIO_ABRIGO)

# Ruido aleatorio (en Soles)
noise = np.random.normal(0, 10 * PRECIO_ABRIGO, n_periods)

# 3. Combinar componentes (en Soles)
sales_soles = trend + seasonality + events + noise
sales_unidades = (sales_soles / PRECIO_ABRIGO).round(1)  # Convertir a unidades

# 4. Crear DataFrame
topitop = pd.DataFrame({
    "Fecha": dates,
    "Ventas_PEN": sales_soles.round(2),
    "Unidades": sales_unidades,
    "Temporada_alta": np.where((dates.month == 12) | (dates.month == 7), 1, 0)
})
topitop.set_index("Fecha", inplace=True)

# Guardar dataset
topitop.to_excel("ventas_topitop.xlsx")