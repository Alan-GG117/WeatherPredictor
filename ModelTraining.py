import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Cargando datos limpios...")
ds = pd.read_csv("Dataset_OpenMeteo_Cleaned.csv")

# Separación de features (x) y target (y)
X = ds[['hora', 'mes', 'dia_del_año', 'temp_hora_anterior']]
y = ds['temperatura']

# División Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Configuración del modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Entrenando modelo... (Esto puede tomar unos segundos)")
model.fit(X_train, y_train)

# Predicción y evaluación
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

print("\n Resultados de la evaluación")
print(f"Error absoluto medio (MAE): {mae:.2f}°C")
print(f"Error cuadrático medio (MSE): {mse:.2f}")

# --- CÓDIGO ACTUALIZADO PARA LA GRÁFICA CON FECHAS ---

# 1. Recuperamos la columna de fechas y la convertimos al formato correcto
ds['time'] = pd.to_datetime(ds['time'])

# 2. Extraemos exactamente las fechas que corresponden al 20% de prueba (Test)
fechas_test = ds['time'].iloc[y_test.index]

# 3. Gráfico de predicción usando las fechas reales en el eje X
plt.figure(figsize=(12,5))

# Usamos fechas_test[:100] en lugar de solo graficar los valores 'y'
plt.plot(fechas_test[:100], y_test.values[:100], label="Temperatura real", color="blue", marker="o", markersize=3)
plt.plot(fechas_test[:100], prediction[:100], label="Predicción del modelo", color="red", linestyle='dashed')

plt.title("Temperatura Real vs Predicción (Primeras 100 horas de prueba)")
plt.xlabel("Fecha y Hora")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)

# Esto rota las fechas en el eje X para que no se encimen y sean legibles
plt.xticks(rotation=45)
plt.tight_layout() # Ajusta los márgenes para que todo quepa perfecto

plt.show()

# --- EXTRA PARA TU COMPARACIÓN CON EL SENSOR ---
# Si quieres ver en la consola en qué fecha y hora exacta empezó la prueba:
print(f"\nLas predicciones de prueba comienzan exactamente el: {fechas_test.iloc[0]}")
