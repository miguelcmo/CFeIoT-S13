import streamlit as st
import streamlit.components.v1 as components

st.title("Dashboard IoT del Clima")

# Inserta el iframe dentro del componente HTML
iframe_code = """
<iframe 
    src="https://miguelcmo.grafana.net/d-solo/aehqn58kr54aof/home-iot-weather-conditions?orgId=1&from=1747328396681&to=1747349996681&timezone=browser&refresh=10s&panelId=3&__feature.dashboardSceneSolo"
    width="450" 
    height="200" 
    frameborder="0">
</iframe>
"""

components.html(iframe_code, height=220)  # deja un poco más de espacio vertical
