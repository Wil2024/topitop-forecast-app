import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Ignorar advertencias de la librería estadística para mantener la interfaz limpia
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Forecasting Multimodelo Topitop", layout="wide")
st.title("🎯 Simulador Multimodelo y Decisiones Estratégicas")
st.markdown("**Caso Práctico:** Venta de Abrigos | **Precio Unitario:** S/ 79.00")

# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
uploaded_file = st.file_uploader("Sube tu archivo histórico (Excel o CSV). Debe contener las columnas: Fecha, Unidades", type=["xlsx", "csv"])

if uploaded_file:
    try:
        # Lectura flexible
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Procesamiento avanzado de la serie temporal (Soporta Data Diaria o Mensual)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values("Fecha")
        df.set_index('Fecha', inplace=True)
        
        # AGRUPACIÓN BI: Convierte data transaccional (diaria) en agregados mensuales.
        df = df.resample('MS').sum()
        df.fillna(0, inplace=True)
        
        # Panel de Configuración
        st.sidebar.header("⚙️ Parámetros del Modelo")
        target = st.sidebar.selectbox("KPI a Pronosticar (Seleccione Unidades):", df.columns)
        h_pronostico = st.sidebar.slider("Meses a proyectar al futuro:", 6, 36, 12)
        
        # Variable de negocio definida para el caso práctico
        PRECIO_ABRIGO = 79.00
        st.divider()
        
        # ==========================================
        # FASE 1: BACKTESTING Y MÉTRICAS (AHORA CON RMSE)
        # ==========================================
        st.subheader("1️⃣ Evaluación de Precisión (Backtesting)")
        st.write("Entrenamos los modelos omitiendo los últimos 12 meses para validar su precisión frente a datos reales.")
        train = df[target].iloc[:-12]
        test = df[target].iloc[-12:]
        
        with st.spinner("Entrenando motores matemáticos..."):
            # ARIMA
            mod_arima_val = ARIMA(train, order=(1,1,1)).fit()
            fc_arima_val = mod_arima_val.forecast(steps=12)
            
            # SARIMA
            mod_sarima_val = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False).fit(disp=False)
            fc_sarima_val = mod_sarima_val.forecast(steps=12)
            
            # Holt-Winters
            mod_hw_val = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            fc_hw_val = mod_hw_val.forecast(steps=12)
            
            # Prophet
            df_p_val = train.reset_index().rename(columns={'Fecha': 'ds', target: 'y'})
            mod_p_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df_p_val)
            fut_val = mod_p_val.make_future_dataframe(periods=12, freq='MS')
            fc_p_val = mod_p_val.predict(fut_val)['yhat'].iloc[-12:].values
            
            # Consolidación de Métricas
            modelos_lista = ["ARIMA", "SARIMA", "Holt-Winters", "Prophet"]
            forecasts_val = [fc_arima_val, fc_sarima_val, fc_hw_val, fc_p_val]
            metricas = []
            for nombre, fc in zip(modelos_lista, forecasts_val):
                mae = mean_absolute_error(test, fc)
                rmse = np.sqrt(mean_squared_error(test, fc))
                mape = mean_absolute_percentage_error(test, fc) * 100
                metricas.append({
                    "Modelo": nombre,
                    "MAE (Unidades)": mae,
                    "RMSE (Unidades)": rmse,
                    "MAPE (%)": mape
                })
                
            df_metricas = pd.DataFrame(metricas).set_index("Modelo")
            st.dataframe(df_metricas.style.highlight_min(subset=["MAPE (%)"], color="lightgreen").format("{:.2f}"))
            st.divider()
            
            # ==========================================
            # FASE 2: PROYECCIÓN FUTURA MULTIMODELO
            # ==========================================
            st.subheader(f"2️⃣ Proyección al Futuro ({h_pronostico} meses)")
            with st.spinner("Proyectando escenarios futuros con el 100% de la data..."):
                serie_full = df[target]
                fc_arima_fut = ARIMA(serie_full, order=(1,1,1)).fit().forecast(steps=h_pronostico)
                fc_sarima_fut = SARIMAX(serie_full, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False).fit(disp=False).forecast(steps=h_pronostico)
                fc_hw_fut = ExponentialSmoothing(serie_full, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(steps=h_pronostico)
                
                # Usamos Prophet para los intervalos de confianza estratégicos
                df_p_full = serie_full.reset_index().rename(columns={'Fecha': 'ds', target: 'y'})
                m_full = Prophet(yearly_seasonality=True, interval_width=0.95).fit(df_p_full)
                fut_full = m_full.make_future_dataframe(periods=h_pronostico, freq='MS')
                forecast_prophet = m_full.predict(fut_full).iloc[-h_pronostico:]
                fc_p_fut = forecast_prophet.set_index('ds')['yhat']
                lim_inf = forecast_prophet.set_index('ds')['yhat_lower']
                lim_sup = forecast_prophet.set_index('ds')['yhat_upper']
                
                # Gráfico Interactivo Comparativo
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=serie_full.index, y=serie_full, name="Histórico Real", line=dict(color='black', width=3)))
                fig.add_trace(go.Scatter(x=fc_arima_fut.index, y=fc_arima_fut, name="ARIMA", line=dict(dash='dot', color='blue')))
                fig.add_trace(go.Scatter(x=fc_sarima_fut.index, y=fc_sarima_fut, name="SARIMA", line=dict(dash='dash', color='red')))
                fig.add_trace(go.Scatter(x=fc_hw_fut.index, y=fc_hw_fut, name="Holt-Winters", line=dict(dash='dashdot', color='green')))
                fig.add_trace(go.Scatter(x=fc_p_fut.index, y=fc_p_fut, name="Prophet", line=dict(dash='longdash', color='purple')))
                fig.update_layout(title="Comparativa de Predicciones", xaxis_title="Fecha", yaxis_title="Unidades de Abrigos")
                
                # ✅ CORREGIDO: use_container_width en lugar de use_container_view
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
                
                # ==========================================
                # FASE 3: DECISIONES ESTRATÉGICAS Y DESCARGA
                # ==========================================
                st.subheader("3️⃣ Panel de Decisiones Estratégicas (Basado en Prophet)")
                st.write("Análisis de Riesgo e Inventario utilizando los Intervalos de Confianza al 95%.")
                
                prediccion_base = np.maximum(0, np.round(fc_p_fut.values))
                escenario_pesimista = np.maximum(0, np.round(lim_inf.values))
                escenario_optimista = np.maximum(0, np.round(lim_sup.values))
                stock_seguridad = escenario_optimista - prediccion_base
                
                ingresos_esperados = prediccion_base * PRECIO_ABRIGO
                riesgo_capital_stock = stock_seguridad * PRECIO_ABRIGO
                
                # Gráfico Estratégico de Bandas
                fig_est = go.Figure()
                fig_est.add_trace(go.Scatter(x=fc_p_fut.index, y=escenario_optimista, name="Límite Superior (Optimista)", line=dict(color='lightblue', width=0)))
                fig_est.add_trace(go.Scatter(x=fc_p_fut.index, y=escenario_pesimista, name="Límite Inferior (Pesimista)", fill='tonexty', fillcolor='rgba(173,216,230,0.4)', line=dict(color='lightblue', width=0)))
                fig_est.add_trace(go.Scatter(x=fc_p_fut.index, y=prediccion_base, name="Pronóstico de Compras (Unidades)", line=dict(color='darkblue', width=3)))
                fig_est.update_layout(title="Rango de Incertidumbre y Plan de Compras de Abrigos", xaxis_title="Fecha", yaxis_title="Unidades a Comprar/Producir")
                
                # ✅ CORREGIDO: use_container_width en lugar de use_container_view
                st.plotly_chart(fig_est, use_container_width=True)
                
                # Consolidación del DataFrame Final
                df_estrategico = pd.DataFrame({
                    "Fecha": fc_p_fut.index.strftime('%Y-%m'),
                    "Pronostico_Base_Unds": prediccion_base,
                    "Escenario_Pesimista_Unds": escenario_pesimista,
                    "Escenario_Optimista_Unds": escenario_optimista,
                    "Stock_Seguridad_Unds": stock_seguridad,
                    "Ingreso_Esperado_PEN": ingresos_esperados,
                    "Capital_Stock_Seguridad_PEN": riesgo_capital_stock
                })
                st.dataframe(df_estrategico.style.format({
                    "Pronostico_Base_Unds": "{:,.0f}",
                    "Escenario_Pesimista_Unds": "{:,.0f}",
                    "Escenario_Optimista_Unds": "{:,.0f}",
                    "Stock_Seguridad_Unds": "{:,.0f}",
                    "Ingreso_Esperado_PEN": "S/. {:,.2f}",
                    "Capital_Stock_Seguridad_PEN": "S/. {:,.2f}"
                }))
                
                # --- EXPORTACIÓN A EXCEL (.xlsx) ---
                st.subheader("📥 Descarga de Plan Comercial (Formato Excel)")
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_estrategico.to_excel(writer, index=False, sheet_name='Plan_Produccion_Finanzas')
                    worksheet = writer.sheets['Plan_Produccion_Finanzas']
                    for i, col in enumerate(df_estrategico.columns):
                        column_len = max(df_estrategico[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, column_len)
                        
                st.download_button(
                    label="📊 Descargar Plan de Producción y Finanzas (Excel)",
                    data=buffer.getvalue(),
                    file_name="plan_estrategico_abrigos_topitop.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"Se encontró un error en la ejecución: {e}. Asegúrese de que el archivo contiene la columna de fechas correcta y seleccione 'Unidades'.")
else:
    st.info("👋 Suba su archivo histórico para iniciar el simulador multimodelo de alta dirección.")

# Footer
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; margin-top: 50px; color: #666;'>
    ©️ 2026 Taller MBA - Modelos de Forecasting y Simulación de Escenarios
    </div>
    """, unsafe_allow_html=True
)
