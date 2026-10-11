"""
Exploración rápida del dataset para ver columnas y países.
"""
import pandas as pd

df = pd.read_csv("data/raw/covid_owid.csv", low_memory=False)

print("=" * 60)
print(f"📊 Dimensiones: {df.shape}")
print("=" * 60)
print("\n📋 TODAS las columnas:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 60)
print("🌍 Países (primeros 30):")
print(df["country"].dropna().unique()[:30])

print("\n" + "=" * 60)
print("📅 Rango de fechas:", df["date"].min(), "→", df["date"].max())

print("\n" + "=" * 60)
print("🇨🇴 Datos de Colombia:")
df_col = df[df["country"] == "Colombia"]
print(f"  Filas: {len(df_col)}")
print(f"  Fechas: {df_col['date'].min()} → {df_col['date'].max()}")
print(f"  Casos totales máx: {df_col['total_cases'].max()}")
print(f"  ¿Tiene new_cases? {'new_cases' in df_col.columns}")
print(f"  Nulos en new_cases: {df_col['new_cases'].isna().sum()}")