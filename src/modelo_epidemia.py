"""
Modelo SEIR basado en agentes para propagación de epidemia.
Usa Mesa + NetworkX para simular contactos entre personas.

Estados:
    S = Susceptible
    E = Expuesto (incubando, no contagia aún)
    I = Infectado (contagia)
    R = Recuperado (inmune)
    V = Vacunado (inmune desde el inicio)
"""

import networkx as nx
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation


# =====================================================================
# AGENTE
# =====================================================================
class Persona(Agent):
    """
    Representa una persona en la población.
    Atributos:
        estado: S, E, I, R o V
        dias_en_estado: contador de días en el estado actual
    """

    def __init__(self, uid, model):
        super().__init__(uid, model)
        self.estado = "S"
        self.dias_en_estado = 0

    def step(self):
        """Actualiza el estado del agente según las reglas SEIR."""

        # --- EXPUESTO: incubando ---
        if self.estado == "E":
            self.dias_en_estado += 1
            if self.dias_en_estado >= self.model.dias_incubacion:
                self.estado = "I"
                self.dias_en_estado = 0

        # --- INFECTADO: contagia a vecinos ---
        elif self.estado == "I":
            self.dias_en_estado += 1

            # Intentar contagiar a cada vecino susceptible
            for vecino_id in self.model.red.neighbors(self.unique_id):
                vecino = self.model.schedule.agents[vecino_id]
                if vecino.estado == "S":
                    if np.random.rand() < self.model.beta:
                        vecino.estado = "E"
                        vecino.dias_en_estado = 0

            # Después de N días, se recupera
            if self.dias_en_estado >= self.model.dias_infeccioso:
                self.estado = "R"
                self.dias_en_estado = 0

        # S, R, V no hacen nada en cada paso


# =====================================================================
# MODELO
# =====================================================================
class ModeloEpidemia(Model):
    """
    Modelo SEIR con red de contactos small-world.

    Parámetros:
        N                  : número de agentes
        beta               : probabilidad de contagio por contacto
        dias_incubacion    : días en estado E antes de pasar a I
        dias_infeccioso    : días en estado I antes de pasar a R
        prob_vacunacion    : fracción inicial de vacunados
        grado_promedio     : número promedio de contactos por persona
        semilla            : semilla aleatoria para reproducibilidad
    """

    def __init__(
        self,
        N=1000,
        beta=0.04,
        dias_incubacion=3,
        dias_infeccioso=7,
        prob_vacunacion=0.0,
        grado_promedio=10,
        semilla=42,
    ):
        super().__init__(seed=semilla)

        self.num_agents = N
        self.beta = beta
        self.dias_incubacion = dias_incubacion
        self.dias_infeccioso = dias_infeccioso
        self.prob_vacunacion = prob_vacunacion
        self.grado_promedio = grado_promedio

        # Programador de eventos
        self.schedule = RandomActivation(self)

        # --- Construir red de contactos (small-world) ---
        # k = grado promedio (debe ser par y < N)
        k = min(grado_promedio, N - 1)
        if k % 2 != 0:
            k += 1  # forzar par
        self.red = nx.watts_strogatz_graph(N, k=k, p=0.1, seed=semilla)

        # --- Crear agentes ---
        for i in range(N):
            agente = Persona(i, self)
            # Vacunar aleatoriamente al inicio
            if np.random.rand() < prob_vacunacion:
                agente.estado = "V"
            self.schedule.add(agente)

        # --- Infectar un pequeño porcentaje al inicio (0.1%) ---
        n_iniciales = max(1, int(N * 0.001))
        # Elegir solo entre los no vacunados
        candidatos = [a.unique_id for a in self.schedule.agents if a.estado == "S"]
        if len(candidatos) > 0:
            n_iniciales = min(n_iniciales, len(candidatos))
            infectados_iniciales = np.random.choice(
                candidatos, n_iniciales, replace=False
            )
            for idx in infectados_iniciales:
                self.schedule.agents[idx].estado = "I"

        # --- Estado inicial del modelo ---
        self.dia = 0
        self.historial = []
        self.registrar_estado()

    # -----------------------------------------------------------------
    # PASO
    # -----------------------------------------------------------------
    def step(self):
        """Avanza un día en la simulación."""
        self.schedule.step()
        self.dia += 1
        self.registrar_estado()

    # -----------------------------------------------------------------
    # REGISTRO
    # -----------------------------------------------------------------
    def registrar_estado(self):
        """Guarda el conteo de agentes por estado en este día."""
        conteo = {"S": 0, "E": 0, "I": 0, "R": 0, "V": 0}
        for agente in self.schedule.agents:
            conteo[agente.estado] += 1

        conteo["dia"] = self.dia
        conteo["activos"] = conteo["I"] + conteo["E"]
        conteo["total_infectados"] = conteo["I"] + conteo["E"] + conteo["R"]
        self.historial.append(conteo)

    # -----------------------------------------------------------------
    # UTILIDADES
    # -----------------------------------------------------------------
    def contar_estado(self, estado):
        """Devuelve cuántos agentes están en un estado dado."""
        return sum(1 for a in self.schedule.agents if a.estado == estado)

    def epidemia_activa(self):
        """True si todavía hay expuestos o infectados."""
        return self.contar_estado("I") + self.contar_estado("E") > 0

    def resumen(self):
        """Devuelve un diccionario con el resumen final de la simulación."""
        pico = max(h["activos"] for h in self.historial)
        dia_pico = next(
            h["dia"] for h in self.historial if h["activos"] == pico
        )
        return {
            "dias_simulados": self.dia,
            "pico_activos": pico,
            "dia_pico": dia_pico,
            "pct_pico": pico / self.num_agents * 100,
            "total_infectados": self.contar_estado("R") + self.contar_estado("I"),
            "susceptibles_finales": self.contar_estado("S"),
            "recuperados": self.contar_estado("R"),
            "vacunados": self.contar_estado("V"),
        }
