import streamlit as st
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración desde archivo local
from config import INFLUX_URL, INFLUX_TOKEN, ORG, BUCKET

# --- Funciones para cargar datos ---
def get_temperature_data():
    query = '''
    from(bucket: "homeiot")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "temperature")
    '''
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df = client.query_api().query_data_frame(org=ORG, query=query)
    df = df[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": "temperatura"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_humidity_data():
    query = '''
    from(bucket: "homeiot")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "humidity")
    '''
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df = client.query_api().query_data_frame(org=ORG, query=query)
    df = df[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": "humedad"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# --- Funciones de detección de anomalías ---
def detectar_anomalias_temperature(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["temperatura"]])
    return df

def detectar_anomalias_humidity(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["humedad"]])
    return df

# --- Streamlit UI ---
st.title("Análisis de sensores ambientales con IA local")

opcion = st.selectbox("Selecciona el tipo de dato a analizar:", ["Temperatura", "Humedad", "Ambos"])

if st.button("Cargar y analizar datos"):
    if opcion in ["Temperatura", "Ambos"]:
        st.header("📈 Análisis de Temperatura")
        df_temp = get_temperature_data()
        st.subheader("Datos crudos:")
        st.dataframe(df_temp)

        st.subheader("Estadísticas descriptivas:")
        st.write(df_temp["temperatura"].describe())

        df_temp = detectar_anomalias_temperature(df_temp)
        outliers_temp = df_temp[df_temp["anomaly"] == -1]

        st.subheader("Visualización con anomalías:")
        fig, ax = plt.subplots()
        sns.lineplot(x="timestamp", y="temperatura", data=df_temp, label="Temperatura", ax=ax)
        ax.scatter(outliers_temp["timestamp"], outliers_temp["temperatura"], color="red", label="Anomalía", zorder=5)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Anomalías detectadas:")
        st.dataframe(outliers_temp)

    if opcion in ["Humedad", "Ambos"]:
        st.header("💧 Análisis de Humedad")
        df_hum = get_humidity_data()
        st.subheader("Datos crudos:")
        st.dataframe(df_hum)

        st.subheader("Estadísticas descriptivas:")
        st.write(df_hum["humedad"].describe())

        df_hum = detectar_anomalias_humidity(df_hum)
        outliers_hum = df_hum[df_hum["anomaly"] == -1]

        st.subheader("Visualización con anomalías:")
        fig, ax = plt.subplots()
        sns.lineplot(x="timestamp", y="humedad", data=df_hum, label="Humedad", ax=ax)
        ax.scatter(outliers_hum["timestamp"], outliers_hum["humedad"], color="red", label="Anomalía", zorder=5)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Anomalías detectadas:")
        st.dataframe(outliers_hum)
