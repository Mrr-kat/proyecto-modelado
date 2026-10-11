"""
Análisis del comportamiento emergente.
Muestra cómo reglas individuales generan patrones colectivos.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modelo_epidemia import ModeloEpidemia

os.makedirs("results/figuras", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)


def analizar_emergente(N=1000, beta=0.045, dias=200):
    """Ejecuta el modelo y analiza el comportamiento emergente."""
    print("🔬 Analizando comportamiento emergente...")

    modelo = ModeloEpidemia(
        N=N, beta=beta, dias_incubacion=3, dias_infeccioso=7,
        prob_vacunacion=0.0, grado_promedio=10, semilla=42
    )

    for _ in range(dias):
        if not modelo.epidemia_activa():
            break
        modelo.step()

    hist = pd.DataFrame(modelo.historial)

    # ---- 1. Curva S (crecimiento acumulado) ----
    hist["acumulados"] = hist["R"] + hist["I"] + hist["E"]
    hist["pct_acumulado"] = hist["acumulados"] / N * 100

    # ---- 2. Velocidad de contagio (nuevos por día) ----
    hist["nuevos"] = hist["acumulados"].diff().fillna(0)

    # ---- 3. R efectivo (aproximado) ----
    # R_t ≈ nuevos(t) / nuevos(t - dias_infeccioso)
    hist["R_efectivo"] = np.nan
    for t in range(7, len(hist)):
        if hist.loc[t-7, "nuevos"] > 0:
            hist.loc[t, "R_efectivo"] = hist.loc[t, "nuevos"] / hist.loc[t-7, "nuevos"]

    # ---- 4. Gráfico 1: Curva S ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(hist["dia"], hist["pct_acumulado"], "b-", lw=2.5)
    axes[0].axhline(y=hist["pct_acumulado"].max(), color="r", ls="--",
                    alpha=0.5, label=f"Pico: {hist['pct_acumulado'].max():.1f}%")
    axes[0].set_xlabel("Días")
    axes[0].set_ylabel("% de población infectada (acumulado)")
    axes[0].set_title("Comportamiento emergente: Curva S de adopción")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Gráfico 2: R efectivo ----
    axes[1].plot(hist["dia"], hist["R_efectivo"], "purple", lw=2)
    axes[1].axhline(y=1.0, color="k", ls="--", alpha=0.7, label="R=1 (umbral)")
    axes[1].set_xlabel("Días")
    axes[1].set_ylabel("R efectivo")
    axes[1].set_title("Número reproductivo efectivo R(t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figuras/emergente_curva_s.png", dpi=150)
    plt.close()

    # ---- 5. Resumen ----
    print(f"\n📊 Comportamiento emergente identificado:")
    print(f"   - Curva S: crecimiento exponencial → desaceleración → meseta")
    print(f"   - Pico de infección: {hist['pct_acumulado'].max():.1f}% de la población")
    print(f"   - R inicial (primeros días): {hist['R_efectivo'].dropna().iloc[:10].mean():.2f}")
    print(f"   - R final (últimos días): {hist['R_efectivo'].dropna().iloc[-10:].mean():.2f}")
    print(f"   - Punto de cruce R=1: día {hist[hist['R_efectivo'] < 1].index[0] if len(hist[hist['R_efectivo'] < 1]) > 0 else 'N/A'}")

    with open("results/tablas/emergente.txt", "w") as f:
        f.write(f"pico_pct={hist['pct_acumulado'].max():.1f}\n")
        f.write(f"R_inicial={hist['R_efectivo'].dropna().iloc[:10].mean():.2f}\n")
        f.write(f"R_final={hist['R_efectivo'].dropna().iloc[-10:].mean():.2f}\n")

    print("✅ Guardado en results/figuras/emergente_curva_s.png")
    return hist


if __name__ == "__main__":
    analizar_emergente()