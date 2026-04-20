import pandas as pd

print("Iniciando procesamiento de datos...")

# 1. Cargar el dataset original
df = pd.read_csv("Dataset_OpenMeteo.csv", skiprows=3)

# 2. Formato y Renombrado
df['time'] = pd.to_datetime(df['time'])
df.rename(columns={'temperature_2m (°C)':'temperatura'}, inplace=True)

# 3. Ingeniería de características
df['hora'] = df['time'].dt.hour
df['mes'] = df['time'].dt.month
df['dia_del_año'] = df['time'].dt.dayofyear
df['temp_hora_anterior'] = df['temperatura'].shift(1)

# 4. Limpieza final (quitar el nulo del inicio)
df = df.dropna()

# 5. Guardar el dataset limpio
outputName = "Dataset_OpenMeteo_Cleaned.csv"
df.to_csv(outputName, index=False)

print(f"Los datos procesados fueron almacenados en {outputName}")
