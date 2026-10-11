"""
Análisis estadístico del dataset COVID-19 (Colombia, primera ola 2020).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("results/figuras", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def cargar_y_filtrar(pais="Colombia", fecha_ini="2020-03-01", fecha_fin="2021-03-31"):
    print(f"📂 Cargando dataset...")
    df = pd.read_csv("data/raw/covid_owid.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    print(f"🔍 Filtrando {pais}...")
    df_pais = df[df["country"] == pais].copy()
    df_pais = df_pais.sort_values("date").reset_index(drop=True)

    # Filtrar rango de fechas
    df_pais = df_pais[
        (df_pais["date"] >= fecha_ini) & (df_pais["date"] <= fecha_fin)
    ].copy()

    # Limpiar
    df_pais["new_cases"] = df_pais["new_cases"].fillna(0).clip(lower=0)
    df_pais["new_deaths"] = df_pais["new_deaths"].fillna(0).clip(lower=0)
    df_pais["population"] = df_pais["population"].ffill().bfill()

    # Media móvil 7 días
    df_pais["new_cases_smooth"] = df_pais["new_cases"].rolling(7, min_periods=1).mean()

    df_pais.to_csv("data/processed/covid_limpio.csv", index=False)
    print(f"✅ Datos limpios guardados: {df_pais.shape}")
    print(f"   Rango: {df_pais['date'].min().date()} → {df_pais['date'].max().date()}")
    return df_pais


def estadisticas_descriptivas(df):
    cols = ["new_cases", "new_deaths", "new_cases_smooth"]
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]

    stats = df[cols].describe().T
    stats["varianza"] = df[cols].var()
    stats["mediana"] = df[cols].median()

    stats = stats[["mean", "mediana", "std", "varianza", "min", "max"]]
    stats.columns = ["Media", "Mediana", "Desv_std", "Varianza", "Min", "Max"]

    stats.to_csv("results/tablas/estadisticas.csv")
    print("\n📊 Estadísticas descriptivas:")
    print(stats.round(2).to_string())
    return stats


def graficos(df, pais="Colombia"):
    # 1. Curva de casos nuevos + suavizado
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["new_cases"], color="lightcoral", lw=1, alpha=0.6, label="Casos diarios")
    ax.plot(df["date"], df["new_cases_smooth"], color="crimson", lw=2, label="Media móvil 7d")
    ax.set_title(f"Casos nuevos diarios de COVID-19 — {pais}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Casos nuevos")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figuras/curva_casos.png", dpi=150)
    plt.close()

    # 2. Histograma
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["new_cases"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribución de casos nuevos diarios")
    ax.set_xlabel("Casos nuevos")
    plt.tight_layout()
    plt.savefig("results/figuras/histograma_casos.png", dpi=150)
    plt.close()

    # 3. Boxplot por mes
    df["mes"] = df["date"].dt.to_period("M").astype(str)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=df, x="mes", y="new_cases", ax=ax, color="skyblue")
    ax.set_title("Casos nuevos por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Casos nuevos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figuras/boxplot_mes.png", dpi=150)
    plt.close()

    print("✅ Gráficos guardados en results/figuras/")


if __name__ == "__main__":
    df = cargar_y_filtrar("Colombia")
    estadisticas_descriptivas(df)
    graficos(df, "Colombia")