"""
Validación mejorada con ajuste de escala temporal.
Compara la dinámica del modelo con la real alineando duraciones.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.modelo_epidemia import ModeloEpidemia

os.makedirs("results/figuras", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)

N_VAL = 1000
DIAS_VAL = 250
N_CORRIDAS = 5


def leer_beta():
    ruta = "results/tablas/beta_optimo.txt"
    if os.path.exists(ruta):
        with open(ruta) as f:
            for linea in f:
                if linea.startswith("beta="):
                    return float(linea.split("=")[1])
    return 0.045


def alinear_curvas(curva_sim, curva_real, n_puntos=200):
    """Reescala ambas curvas a un eje X común [0,1]."""
    x_sim = np.linspace(0, 1, len(curva_sim))
    x_real = np.linspace(0, 1, len(curva_real))

    f_sim = interp1d(x_sim, curva_sim, kind="linear")
    f_real = interp1d(x_real, curva_real, kind="linear")

    x_comun = np.linspace(0, 1, n_puntos)
    sim_interp = f_sim(x_comun)
    real_interp = f_real(x_comun)

    sim_interp = sim_interp / (sim_interp.max() + 1e-9)
    real_interp = real_interp / (real_interp.max() + 1e-9)
    return x_comun, sim_interp, real_interp


def suavizar(curva, ventana=15):
    """Suaviza con media móvil."""
    return pd.Series(curva).rolling(ventana, min_periods=1, center=True).mean().values


def validar(beta=None, N=N_VAL, dias=DIAS_VAL, n_corridas=N_CORRIDAS):
    if beta is None:
        beta = leer_beta()

    print(f"🔬 Validación mejorada con beta = {beta:.4f}")
    print(f"   Config: N={N}, días={dias}, corridas={n_corridas}\n")

    inicio = time.time()

    # ---- 1. Corridas ----
    curvas = []
    for i in range(n_corridas):
        modelo = ModeloEpidemia(
            N=N, beta=beta, dias_incubacion=3, dias_infeccioso=7,
            prob_vacunacion=0.0, grado_promedio=10, semilla=i
        )
        for _ in range(dias):
            if not modelo.epidemia_activa():
                break
            modelo.step()
        hist = pd.DataFrame(modelo.historial)
        curvas.append(hist["activos"].values)
        print(f"   Corrida {i+1}/{n_corridas}: {modelo.dia} días, "
              f"pico {max(hist['activos'])}")

    # ---- 2. Promedio y suavizado ----
    max_len = max(len(c) for c in curvas)
    curvas_pad = [np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curvas]
    promedio = np.mean(curvas_pad, axis=0)
    promedio_suave = suavizar(promedio, ventana=10)

    # ---- 3. Curva real ----
    real_df = pd.read_csv("data/processed/covid_limpio.csv")
    real_vals = real_df["new_cases_smooth"].values
    real_suave = suavizar(real_vals, ventana=15)

    # ---- 4. Alinear ----
    x_comun, sim_norm, real_norm = alinear_curvas(promedio_suave, real_suave, n_puntos=200)

    # ---- 5. Gráfico ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Izquierda: alineadas
    axes[0].plot(x_comun, real_norm, "k-", lw=2.5, label="Real (Colombia)")
    axes[0].plot(x_comun, sim_norm, "r--", lw=2, label="Modelo ABM")
    axes[0].fill_between(
        x_comun,
        np.clip(sim_norm - 0.08, 0, None),
        np.clip(sim_norm + 0.08, 0, 1),
        color="red", alpha=0.15, label="±ruido"
    )
    axes[0].set_xlabel("Tiempo normalizado (0 → 1)")
    axes[0].set_ylabel("Casos activos (normalizado)")
    axes[0].set_title(f"Validación alineada\n(beta={beta:.4f}, {n_corridas} corridas)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Derecha: escala original
    axes[1].plot(range(len(real_vals)), real_vals, "k-", lw=2, label="Real (Colombia)")
    axes[1].plot(range(len(promedio)), promedio, "r--", lw=2, label="Modelo ABM")
    axes[1].set_xlabel("Días")
    axes[1].set_ylabel("Casos activos")
    axes[1].set_title("Comparación en escala original")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figuras/validacion.png", dpi=150)
    plt.close()

    # ---- 6. Métricas ----
    mse = np.mean((sim_norm - real_norm) ** 2)
    mae = np.mean(np.abs(sim_norm - real_norm))
    corr = np.corrcoef(sim_norm, real_norm)[0, 1]

    # Correlación de forma (sobre las derivadas, más robusta)
    d_sim = np.diff(sim_norm)
    d_real = np.diff(real_norm)
    corr_forma = np.corrcoef(d_sim, d_real)[0, 1]

    print(f"\n{'='*55}")
    print(f"📊 MÉTRICAS DE VALIDACIÓN")
    print(f"{'='*55}")
    print(f"   MSE:            {mse:.4f}")
    print(f"   MAE:            {mae:.4f}")
    print(f"   Correlación:    {corr:.4f}")
    print(f"   Corr. de forma: {corr_forma:.4f}  (robusta a desfase)")
    print(f"   Tiempo:         {time.time()-inicio:.1f}s")
    print(f"{'='*55}")

    if mse < 0.08 and corr_forma > 0.3:
        print("   ✅ Validación BUENA")
    elif mse < 0.15:
        print("   ✅ Validación ACEPTABLE")
    else:
        print("   ⚠️  Validación DÉBIL — revisar en diapositivas como limitación")

    with open("results/tablas/validacion.txt", "w") as f:
        f.write(f"beta={beta:.4f}\n")
        f.write(f"MSE={mse:.4f}\n")
        f.write(f"MAE={mae:.4f}\n")
        f.write(f"correlacion={corr:.4f}\n")
        f.write(f"correlacion_forma={corr_forma:.4f}\n")

    return mse, mae, corr, corr_forma


if __name__ == "__main__":
    validar()