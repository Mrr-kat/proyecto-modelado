"""
Descarga el dataset COVID-19 de Our World in Data.
"""
import os
import requests
import pandas as pd
from tqdm import tqdm

URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
RUTA_RAW = "data/raw/covid_owid.csv"

def descargar_dataset():
    os.makedirs("data/raw", exist_ok=True)

    print("📥 Descargando dataset COVID-19 de OWID...")
    respuesta = requests.get(URL, stream=True)
    total = int(respuesta.headers.get("content-length", 0))

    with open(RUTA_RAW, "wb") as f, tqdm(
        desc=RUTA_RAW, total=total, unit="B", unit_scale=True
    ) as barra:
        for chunk in respuesta.iter_content(chunk_size=8192):
            f.write(chunk)
            barra.update(len(chunk))

    print(f"✅ Dataset guardado en: {RUTA_RAW}")
    return RUTA_RAW


def explorar_dataset(ruta=RUTA_RAW):
    df = pd.read_csv(ruta)
    print(f"\n📊 Dimensiones: {df.shape}")
    print(f"📋 Columnas: {list(df.columns)[:20]}...")
    print(f"\n🌍 Países disponibles: {df['country'].nunique()}")
    print(f"📅 Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    return df


if __name__ == "__main__":
    descargar_dataset()
    df = explorar_dataset()
    print(df.head())