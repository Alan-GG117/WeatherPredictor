import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Cargando datos limpios...")
ds = pd.read_csv("Dataset_OpenMeteo_Cleaned.csv")

# 1. Asegurar que el tiempo está en el formato correcto
ds['time'] = pd.to_datetime(ds['time'])

courtDate = input("Ingresa tu fecha de corte, considera: *FORMATO: YYYY-MM-DD*, *ULTIMA FECHA RESGISTRADA EN EL BANCO DE DATOS: 2026-04-17* \n")

# 2. Entrenar antes de la fecha de corte
trainData = ds[ds['time'] < courtDate]
# 3. Probar después de la fecha de corte
testData = ds[ds['time'] >= courtDate]

# Separar Features (x) y Target (y) para entrenamiento
X_train = trainData[['hora', 'mes', 'dia_del_año', 'temp_hora_anterior']]
y_train = trainData['temperatura']

# Separar Features (x) y Target (y) para Prueba
X_test = testData[['hora', 'mes', 'dia_del_año', 'temp_hora_anterior']]
y_test = testData['temperatura']

# 4. Configuración del modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
print(f"Entrenando modelo... (antes de {courtDate})")
model.fit(X_train, y_train)

# 5. Predicción y evaluación
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

print("\n --- Resultados de la evaluación ---")
print(f"Inicio de predicciones: {courtDate}")
print(f"Error absoluto medio: {mae:.2f}°C")
print(f"Erro cuadrático medio: {mse:.2f}")

# 6. Extracción de fechas para el gráfico
plotDays = testData['time'].values

# Gráfico de predicción
plt.figure(figsize=(12,5))
plt.plot(plotDays[:100], y_test.values[:100],
         label="Temperatura real (Open-Meteo)",
         color="blue", marker="o", markersize=3)
plt.plot(plotDays[:100], prediction[:100],
         label="Predicción del modelo (RandomForest)",
         color="red", linestyle='dashed')
plt.title(f"Predicción de temperatura desde {courtDate}")
plt.xlabel("Fecha y hora")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()