"""
Calibración RÁPIDA de beta.
Usa búsqueda en grilla + refinamiento (mucho más rápido que minimize_scalar).
Reduce N y días para acelerar la simulación.
"""
import os
import time
import numpy as np
import pandas as pd
from src.modelo_epidemia import ModeloEpidemia


# ---- Parámetros de la calibración (reducidos para velocidad) ----
N_CALIB = 1000       # antes 5000 → 5x más rápido
DIAS_CALIB = 200     # antes 400 → 2x más rápido


def curva_real():
    """Carga la curva real suavizada de Colombia."""
    df = pd.read_csv("data/processed/covid_limpio.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df["new_cases_smooth"].values


def simular_curva(beta, N=N_CALIB, dias=DIAS_CALIB, semilla=42):
    """Simula el modelo y devuelve la curva de activos."""
    modelo = ModeloEpidemia(
        N=N,
        beta=beta,
        dias_incubacion=3,
        dias_infeccioso=7,
        prob_vacunacion=0.0,
        grado_promedio=10,
        semilla=semilla,
    )
    for _ in range(dias):
        if not modelo.epidemia_activa():
            break
        modelo.step()

    hist = pd.DataFrame(modelo.historial)
    return hist["activos"].values


def error(beta, real_norm):
    """MSE entre curva simulada y real (normalizadas)."""
    sim = simular_curva(beta)
    n = min(len(sim), len(real_norm))
    if n < 10:
        return 1e9
    sim_norm = sim[:n] / (sim[:n].max() + 1e-9)
    real_n = real_norm[:n] / (real_norm[:n].max() + 1e-9)
    return float(np.mean((sim_norm - real_n) ** 2))


def calibrar_rapido():
    os.makedirs("results/tablas", exist_ok=True)

    print("📂 Cargando curva real...")
    real = curva_real()
    real_norm = real / (real.max() + 1e-9)
    print(f"   Días reales: {len(real)}")
    print(f"   Pico real: {real.max():.0f} casos/día")
    print(f"\n⚙️  Configuración rápida:")
    print(f"   N = {N_CALIB} agentes")
    print(f"   Días máx = {DIAS_CALIB}")
    print(f"   Búsqueda: grilla gruesa + refinamiento\n")

    inicio = time.time()

    # ---- FASE 1: Grilla gruesa ----
    print("🔍 FASE 1: Exploración gruesa de beta")
    betas_gruesos = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    errores = {}

    for b in betas_gruesos:
        t0 = time.time()
        e = error(b, real_norm)
        errores[b] = e
        print(f"   beta={b:.3f} → MSE={e:.5f}  ({time.time()-t0:.1f}s)")

    mejor_b = min(errores, key=errores.get)
    print(f"\n   ✅ Mejor beta grueso: {mejor_b:.3f} (MSE={errores[mejor_b]:.5f})")

    # ---- FASE 2: Refinamiento fino ----
    print(f"\n🔬 FASE 2: Refinamiento alrededor de {mejor_b:.3f}")
    paso = 0.005
    betas_finos = [
        max(0.005, mejor_b - paso),
        mejor_b,
        mejor_b + paso,
    ]

    for b in betas_finos:
        if b in errores:
            continue
        t0 = time.time()
        e = error(b, real_norm)
        errores[b] = e
        print(f"   beta={b:.4f} → MSE={e:.5f}  ({time.time()-t0:.1f}s)")

    beta_opt = min(errores, key=errores.get)
    mse_opt = errores[beta_opt]

    print(f"\n{'='*50}")
    print(f"✅ Beta óptimo: {beta_opt:.4f}")
    print(f"   MSE: {mse_opt:.6f}")
    print(f"   Tiempo total: {time.time()-inicio:.1f} s")
    print(f"{'='*50}")

    with open("results/tablas/beta_optimo.txt", "w") as f:
        f.write(f"beta={beta_opt:.4f}\nMSE={mse_opt:.6f}\n")
    print("💾 Guardado en results/tablas/beta_optimo.txt")

    # Guardar tabla completa de la búsqueda
    df_res = pd.DataFrame(
        [{"beta": b, "MSE": e} for b, e in sorted(errores.items())]
    )
    df_res.to_csv("results/tablas/calibracion_busqueda.csv", index=False)

    return beta_opt


if __name__ == "__main__":
    calibrar_rapido()