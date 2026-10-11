"""
Prueba rápida del modelo SEIR.
"""
from src.modelo_epidemia import ModeloEpidemia

print("🧬 Creando modelo...")
modelo = ModeloEpidemia(
    N=2000,
    beta=0.04,
    dias_incubacion=3,
    dias_infeccioso=7,
    semilla=42,
)

print("▶️  Simulando...")
for dia in range(365):
    if not modelo.epidemia_activa():
        break
    modelo.step()

resumen = modelo.resumen()
print("\n" + "=" * 50)
print("📊 RESUMEN DE LA SIMULACIÓN")
print("=" * 50)
for k, v in resumen.items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:.2f}")
    else:
        print(f"  {k:25s}: {v}")
print("=" * 50)