import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

# Configurar la página
st.set_page_config(page_title="Pronóstico TOPITOP", layout="wide")
st.title("🔮 Pronóstico Ventas 2025-2026 - TOPITOP")

# 1. Cargar datos desde Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type="xlsx")
if uploaded_file:
    data = pd.read_excel(uploaded_file, parse_dates=["Fecha"], index_col="Fecha")
    data = data.asfreq("M")
    
    # Seleccionar variable a pronosticar
    variable = st.radio(
        "Variable a pronosticar:",
        options=["Ventas_PEN", "Unidades"],
        horizontal=True
    )
    
    # Dividir datos en train (hasta 2023) y test (2024)
    train = data[data.index.year < 2024][variable]
    test = data[data.index.year == 2024][variable]
    
    # 2. Comparación de Modelos (Validación 2024)
    if st.button("Comparar Modelos en 2024"):
        modelos = {
            "ARIMA": {"modelo": ARIMA(train, order=(1,1,1)), "forecast": None},
            "SARIMA": {"modelo": SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)), "forecast": None},
            "Holt-Winters": {"modelo": ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12), "forecast": None}
        }
        
        # Entrenar y pronosticar
        for nombre in modelos.keys():
            try:
                with st.spinner(f"Entrenando {nombre}..."):
                    if nombre == "Holt-Winters":
                        fit_model = modelos[nombre]["modelo"].fit()
                        forecast = fit_model.forecast(len(test))
                    elif nombre == "SARIMA":
                        fit_model = modelos[nombre]["modelo"].fit(disp=False)
                        forecast = fit_model.get_forecast(steps=len(test)).predicted_mean
                    else:
                        fit_model = modelos[nombre]["modelo"].fit()
                        forecast = fit_model.forecast(steps=len(test))
                    
                    modelos[nombre]["forecast"] = forecast
                    modelos[nombre]["mae"] = mean_absolute_error(test, forecast)
                    modelos[nombre]["rmse"] = np.sqrt(mean_squared_error(test, forecast))
                
                st.session_state[nombre] = modelos[nombre]
            except Exception as e:
                st.error(f"Error en {nombre}: {str(e)}")
        
        # Mostrar métricas
        st.subheader("🔍 Rendimiento en 2024 (MAE/RMSE)")
        cols = st.columns(3)
        for i, nombre in enumerate(["ARIMA", "SARIMA", "Holt-Winters"]):
            if nombre in st.session_state:
                with cols[i]:
                    st.markdown(f"**{nombre}**")
                    st.metric("MAE", f"{st.session_state[nombre]['mae']:.2f}")
                    st.metric("RMSE", f"{st.session_state[nombre]['rmse']:.2f}")
        
        # Gráfico comparativo
        st.subheader("📊 Pronósticos vs Realidad (2024)")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(train.index, train, label="Entrenamiento (2020-2023)", color="blue")
        ax.plot(test.index, test, label="Real 2024", color="green", linewidth=2)
        for nombre in ["ARIMA", "SARIMA", "Holt-Winters"]:
            if nombre in st.session_state:
                ax.plot(test.index, st.session_state[nombre]["forecast"], label=f"{nombre}", linestyle="--")
        ax.set_title("Comparación de Modelos - 2024", fontweight="bold")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # 3. Pronóstico 2025-2026 con modelo seleccionado
    st.sidebar.header("Configuración de Pronóstico")
    modelo_seleccionado = st.sidebar.selectbox(
        "Elige modelo:",
        ["ARIMA", "SARIMA", "Holt-Winters"]
    )
    
    años_pronostico = st.sidebar.slider("Años a pronosticar:", 1, 2, 1)
    
    if st.sidebar.button("Generar Pronóstico"):
        if "ARIMA" not in st.session_state:
            st.error("⚠️ Primero compara los modelos en 2024!")
        else:
            try:
                # Configuración
                meses_pronostico = años_pronostico * 12
                serie_full = data[variable]
                
                # Entrenar modelo
                if modelo_seleccionado == "Holt-Winters":
                    modelo = ExponentialSmoothing(serie_full, trend="add", seasonal="add", seasonal_periods=12)
                    resultados = modelo.fit()
                    forecast = resultados.forecast(meses_pronostico)
                    # Holt-Winters no provee intervalos de confianza
                    pronostico_ci = pd.DataFrame({
                        "lower": forecast * 0.9,  # Simulación del 10%
                        "upper": forecast * 1.1
                    })
                elif modelo_seleccionado == "SARIMA":
                    modelo = SARIMAX(serie_full, order=(1,1,1), seasonal_order=(1,1,1,12))
                    resultados = modelo.fit(disp=False)
                    forecast_obj = resultados.get_forecast(steps=meses_pronostico)
                    forecast = forecast_obj.predicted_mean
                    pronostico_ci = forecast_obj.conf_int()
                else:
                    modelo = ARIMA(serie_full, order=(1,1,1))
                    resultados = modelo.fit()
                    forecast_obj = resultados.get_forecast(steps=meses_pronostico)
                    forecast = forecast_obj.predicted_mean
                    pronostico_ci = forecast_obj.conf_int()
                
                # Generar fechas futuras
                ultima_fecha = serie_full.index[-1]
                fechas_futuras = [ultima_fecha + DateOffset(months=i) for i in range(1, meses_pronostico+1)]
                fechas_futuras = pd.DatetimeIndex(fechas_futuras)
                
                # Stock de seguridad
                mae = st.session_state[modelo_seleccionado]["mae"]
                stock_seguridad = np.ceil(mae * 1.5).astype(int)
                
                # Gráfico
                st.subheader(f"🚀 Pronóstico {fechas_futuras[0].year}-{fechas_futuras[-1].year} ({modelo_seleccionado})")
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Histórico
                ax.plot(serie_full.index, serie_full, label="Histórico 2020-2024", color="blue")
                
                # Pronóstico e intervalo
                ax.plot(fechas_futuras, forecast, label="Pronóstico", color="red", linestyle="--")
                ax.fill_between(
                    fechas_futuras,
                    pronostico_ci.iloc[:, 0],
                    pronostico_ci.iloc[:, 1],
                    color="pink",
                    alpha=0.3,
                    label="Intervalo 95% Confianza"
                )
                
                # Stock de seguridad
                ax.axhline(y=stock_seguridad, color="green", linestyle=":", 
                          label=f"Stock Seguridad: {stock_seguridad}")
                
                ax.set_title(f"Pronóstico {fechas_futuras[0].year}-{fechas_futuras[-1].year} | MAE: {mae:.2f}", 
                            fontweight="bold")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Unidades" if variable == "Unidades" else "Ventas (PEN)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                # Datos para descargar
                df_export = pd.DataFrame({
                    "Fecha": fechas_futuras,
                    "Pronóstico": forecast,
                    "Límite Inferior": pronostico_ci.iloc[:, 0],
                    "Límite Superior": pronostico_ci.iloc[:, 1],
                    "Stock Seguridad": stock_seguridad
                })
                
                # Botón de descarga
                st.download_button(
                    label="📥 Descargar Pronóstico (Excel)",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"pronostico_{modelo_seleccionado}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("👋 Sube el archivo 'ventas_topitop.xlsx' para comenzar.")


# Footer con autoría y derechos (agregar al final del script)
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; margin-top: 50px; color: #666;'>
        ©️ 2024 Desarrollado por <b>Wilton Torvisco</b> | 
        <a href='https://github.com/Wil2024' target='_blank'>GitHub</a> | 
        Todos los derechos reservados.
    </div>
    """,
    unsafe_allow_html=True
)