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

# Gráfico de predicción
plt.figure(figsize=(12,5))
plt.plot(y_test.values[:100], label="Temperatura real", color="blue", marker="o", markersize=3)
plt.plot(prediction[:100], label="Predicción del modelo", color="red", linestyle='dashed')
plt.title("Temperatura Real vs Predicción (Primeras 100 horas del test)")
plt.xlabel("Horas")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.show()
