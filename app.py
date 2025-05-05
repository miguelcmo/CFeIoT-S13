import streamlit as st
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración desde archivo local
from config import INFLUX_URL, INFLUX_TOKEN, ORG, BUCKET

# --- Cargar datos desde InfluxDB ---
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

# --- Cargar datos desde InfluxDB ---
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

# --- Detección de anomalías con Isolation Forest ---
def detectar_anomalias_temperature(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["temperatura"]])
    return df

# --- Detección de anomalías con Isolation Forest ---
def detectar_anomalias_humidity(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["humedad"]])
    return df

# --- Streamlit UI Temperature ---
st.title("Análisis de Temperatura con IA local")

if st.button("Cargar y analizar datos de temperatrura"):
    df = get_temperature_data()
    st.subheader("Datos crudos:")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas:")
    st.write(df["temperatura"].describe())

    df = detectar_anomalias_temperature(df)
    outliers = df[df["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="temperatura", data=df, label="Temperatura", ax=ax)
    ax.scatter(outliers["timestamp"], outliers["temperatura"], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)

# --- Streamlit UI Humidity ---
if st.button("Cargar y analizar datos de humedad"):
    df = get_humidity_data()
    st.subheader("Datos crudos:")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas:")
    st.write(df["humedad"].describe())

    df = detectar_anomalias_humidity(df)
    outliers = df[df["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="humedad", data=df, label="Humedad", ax=ax)
    ax.scatter(outliers["timestamp"], outliers["humedad"], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)
