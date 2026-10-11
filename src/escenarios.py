"""
Ejecución de la situación base + 3 escenarios alternativos.
La BASE es la referencia (situación actual), NO es un escenario.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modelo_epidemia import ModeloEpidemia

os.makedirs("results/figuras", exist_ok=True)
os.makedirs("results/tablas", exist_ok=True)

# ---- Configuración ----
N_ESC = 1000
DIAS_ESC = 200
N_CORRIDAS = 10
BETA_BASE = 0.045

# ---- BASE: situación actual (sin intervención) ----
BASE = {
    "beta": BETA_BASE,
    "vacunacion": 0.0,
    "cuarentena": False,
}

# ---- 3 ESCENARIOS ALTERNATIVOS ----
ESCENARIOS = {
    "Esc1 (vacunación 30%)": {
        "beta": BETA_BASE,
        "vacunacion": 0.30,
        "cuarentena": False,
    },
    "Esc2 (cuarentena al 1%)": {
        "beta": BETA_BASE,
        "vacunacion": 0.0,
        "cuarentena": True,
    },
    "Esc3 (vacunación + cuarentena)": {
        "beta": BETA_BASE,
        "vacunacion": 0.30,
        "cuarentena": True,
    },
}


def ejecutar_config(nombre, config, n_corridas=N_CORRIDAS):
    """Ejecuta una configuración n_corridas veces."""
    resultados = []
    curvas = []

    for i in range(n_corridas):
        modelo = ModeloEpidemia(
            N=N_ESC,
            beta=config["beta"],
            dias_incubacion=3,
            dias_infeccioso=7,
            prob_vacunacion=config["vacunacion"],
            grado_promedio=10,
            semilla=i,
        )

        pico = 0
        dia_pico = 0
        umbral_cuarentena = N_ESC * 0.01

        for dia in range(DIAS_ESC):
            if not modelo.epidemia_activa():
                break
            modelo.step()

            activos = modelo.contar_estado("I") + modelo.contar_estado("E")
            if activos > pico:
                pico = activos
                dia_pico = dia

            if config["cuarentena"] and activos > umbral_cuarentena:
                modelo.beta = config["beta"] * 0.4

        hist = pd.DataFrame(modelo.historial)
        curvas.append(hist["activos"].values)

        total_inf = modelo.contar_estado("R") + modelo.contar_estado("I")
        resultados.append({
            "pico": pico,
            "dia_pico": dia_pico,
            "total_infectados": total_inf,
            "duracion": modelo.dia,
            "pct_infectados": total_inf / N_ESC * 100,
        })

    return pd.DataFrame(resultados), curvas


def main():
    print("=" * 60)
    print("🎯 EJECUCIÓN: BASE + 3 ESCENARIOS")
    print("=" * 60)
    print(f"Config: N={N_ESC}, días={DIAS_ESC}, corridas={N_CORRIDAS}")
    print(f"Beta base: {BETA_BASE}\n")

    inicio = time.time()
    resumen = []
    todas_curvas = {}
    picos_por_caso = {}

    # ---- 1. Situación BASE ----
    print("▶️  SITUACIÓN BASE (sin intervención)")
    t0 = time.time()
    df_base, curvas_base = ejecutar_config("Base", BASE)
    todas_curvas["Base (situación actual)"] = curvas_base
    picos_por_caso["Base"] = df_base["pico"].values

    resumen.append({
        "Escenario": "BASE (situación actual)",
        "Pico (media)": df_base["pico"].mean(),
        "Pico (std)": df_base["pico"].std(),
        "Día pico": df_base["dia_pico"].mean(),
        "Total infectados": df_base["total_infectados"].mean(),
        "% infectados": df_base["pct_infectados"].mean(),
        "Duración (días)": df_base["duracion"].mean(),
    })
    print(f"   ✅ Pico medio: {df_base['pico'].mean():.0f} ± "
          f"{df_base['pico'].std():.0f} ({time.time()-t0:.1f}s)")

    # ---- 2. Tres ESCENARIOS alternativos ----
    for nombre, config in ESCENARIOS.items():
        print(f"\n▶️  {nombre}")
        t0 = time.time()
        df, curvas = ejecutar_config(nombre, config)
        todas_curvas[nombre] = curvas
        picos_por_caso[nombre] = df["pico"].values

        resumen.append({
            "Escenario": nombre,
            "Pico (media)": df["pico"].mean(),
            "Pico (std)": df["pico"].std(),
            "Día pico": df["dia_pico"].mean(),
            "Total infectados": df["total_infectados"].mean(),
            "% infectados": df["pct_infectados"].mean(),
            "Duración (días)": df["duracion"].mean(),
        })
        print(f"   ✅ Pico medio: {df['pico'].mean():.0f} ± "
              f"{df['pico'].std():.0f} ({time.time()-t0:.1f}s)")

    # ---- Tabla resumen ----
    tabla = pd.DataFrame(resumen).round(2)
    tabla.to_csv("results/tablas/escenarios.csv", index=False)

    # Calcular reducción porcentual respecto a base
    pico_base = tabla.loc[0, "Pico (media)"]
    tabla["Reducción vs base (%)"] = (
        (pico_base - tabla["Pico (media)"]) / pico_base * 100
    ).round(1)

    print(f"\n{'='*70}")
    print("📊 RESUMEN: BASE + 3 ESCENARIOS")
    print(f"{'='*70}")
    print(tabla.to_string(index=False))
    print(f"\n⏱️  Tiempo total: {time.time()-inicio:.1f}s")

    # ---- Gráfico 1: Barras comparativas ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tabla))
    colores = ["#7f7f7f", "#2ca02c", "#1f77b4", "#9467bd"]  # gris para base
    bars = ax.bar(x, tabla["Pico (media)"],
                   yerr=tabla["Pico (std)"], capsize=6,
                   color=colores, edgecolor="black", lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(tabla["Escenario"], rotation=15, ha="right")
    ax.set_ylabel("Pico de infectados activos")
    ax.set_title(f"Comparación: situación base vs. 3 escenarios\n"
                 f"({N_CORRIDAS} corridas por caso)")
    for bar, val in zip(bars, tabla["Pico (media)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figuras/escenarios_pico.png", dpi=150)
    plt.close()

    # ---- Gráfico 2: Boxplot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(picos_por_caso.values(),
               labels=list(picos_por_caso.keys()),
               patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Pico de infectados")
    ax.set_title("Distribución del pico: base vs. 3 escenarios")
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figuras/escenarios_boxplot.png", dpi=150)
    plt.close()

    # ---- Gráfico 3: Curvas promedio ----
    fig, ax = plt.subplots(figsize=(12, 6))
    colores_curva = ["#7f7f7f", "#2ca02c", "#1f77b4", "#9467bd"]
    for (nombre, curvas), color in zip(todas_curvas.items(), colores_curva):
        max_len = max(len(c) for c in curvas)
        curvas_pad = [np.pad(c, (0, max_len-len(c)), constant_values=0) for c in curvas]
        promedio = np.mean(curvas_pad, axis=0)
        lw = 3 if "Base" in nombre else 2
        ax.plot(promedio, lw=lw, label=nombre, color=color)

    ax.set_xlabel("Días desde el primer caso")
    ax.set_ylabel("Infectados activos")
    ax.set_title("Curvas promedio: base vs. 3 escenarios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figuras/escenarios_curvas.png", dpi=150)
    plt.close()

    print("\n✅ Archivos generados:")
    print("   - results/tablas/escenarios.csv")
    print("   - results/figuras/escenarios_pico.png")
    print("   - results/figuras/escenarios_boxplot.png")
    print("   - results/figuras/escenarios_curvas.png")


if __name__ == "__main__":
    main()