import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def modelTraining(csv):
    ds = pd.read_csv(csv)
    X = ds[['hora', 'mes', 'dia_del_año', 'temp_hora_anterior']]
    y = ds['temperatura']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def getCurrentWeather(latitud=19.4285, longitud=-99.1277):
    # Cambié el %2F por una diagonal normal por si la librería requests se estaba confundiendo
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitud}&longitude={longitud}&current_weather=true&timezone=America/Mexico_City"

    print(f"-> Consultando URL: {url}")
    respuesta = requests.get(url)

    if respuesta.status_code != 200:
        print(f"Error de conexión. El servidor respondió con código: {respuesta.status_code}")
        print(f"Mensaje del servidor: {respuesta.text}")
        # Si falla, devolvemos un valor por defecto para que no truene tu programa
        return 16.0, pd.to_datetime('today')

    try:
        datos = respuesta.json()
        temp_actual = datos['current_weather']['temperature']
        hora_actual = pd.to_datetime(datos['current_weather']['time'])
        return temp_actual, hora_actual
    except Exception as e:
        print(f"Error al decodificar JSON. Respuesta cruda: {respuesta.text}")
        return 16.0, pd.to_datetime('today')

def predictFuture(model, baseTemp, baseHour, predictHours=24):
    predictions = []
    dates = []
    previousTemp = baseTemp
    simulatedHour = baseHour

    for _ in range(predictHours):
        simulatedHour += timedelta(hours=1)

        X_future = pd.DataFrame([{
            'hora': simulatedHour.hour,
            'mes': simulatedHour.month,
            'dia_del_año': simulatedHour.dayofyear,
            'temp_hora_anterior': previousTemp,
        }])

        predictedTemp = model.predict(X_future)[0]
        predictions.append(predictedTemp)
        dates.append(simulatedHour)

        previousTemp = predictedTemp

    return dates, predictions

def plotResults(dates, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predictions, marker='o', color='forestgreen', linestyle='-', linewidth=2)

    for i in range (len(dates)):
        plt.text(dates[i], predictions[i] + 0.2, f'{predictions[i]:.1f}°',
                 ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    plt.title(f"Predicción del Clima - Próximas {len(dates)} horas\n(Iniciando el {dates[0].strftime('%Y-%m-%d')})")
    plt.xlabel("Hora del día")
    plt.ylabel("Temperatura (°C)")

    plt.xticks(dates, [f.strftime('%H:00') for f in dates], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Bloque principal de código
if __name__ == "__main__":
    archivo_datos = "Dataset_OpenMeteo_Cleaned.csv"

    print("\n1. Entrenando modelo definitivo con el 100% de los datos...")
    modelo_rf = modelTraining(archivo_datos)

    print("2. Obteniendo clima actual de la API...")
    temp_hoy, hora_hoy = getCurrentWeather()
    print(f"   -> Temperatura base obtenida: {temp_hoy}°C a las {hora_hoy.strftime('%H:%M')}")

    print("\n3. Generando pronóstico para las próximas 24 horas...")
    fechas_futuras, temps_futuras = predictFuture(modelo_rf, temp_hoy, hora_hoy)

    print("4. Abriendo gráfica de resultados...")
    plotResults(fechas_futuras, temps_futuras)
    print("\n¡Práctica ejecutada con éxito!")