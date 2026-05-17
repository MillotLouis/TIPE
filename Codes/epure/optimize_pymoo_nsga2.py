from __future__ import annotations

import random
from dataclasses import replace
from statistics import mean

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from simulation_hello import SimConfig, ProtocolConfig, Simulation


BASE_CONF = SimConfig(
    nb_nodes=40,
    area_size=800,
    max_dist=250,
    init_bat=100,
    conso=(0.0082, 0.00164, 10),
    dt=0.25,
    ttl_max=7,
    seuil_coeff=0.075,
    coeff_dist_weight=0.6,
    coeff_bat_weight=0.4,
    duration=100,

    # nouveaux paramètres optimisables
    x_min=0.15,
    x_safe=0.80,
    p_short=2.0,
    p_long=2.0,
    p_bat=2.0,
    critical_bat_penalty=2.0,
)

TRACE_FILES = [
    r"C:\Users\millo\Documents\GitHub\TIPE\bm_files\40rw0.movements",
    r"C:\Users\millo\Documents\GitHub\TIPE\bm_files\40rw1.movements",
    r"C:\Users\millo\Documents\GitHub\TIPE\bm_files\40rw2.movements",
    r"C:\Users\millo\Documents\GitHub\TIPE\bm_files\40rw3.movements",
    r"C:\Users\millo\Documents\GitHub\TIPE\bm_files\40rw4.movements",
]

SEEDS = [10_000 + i for i in range(len(TRACE_FILES))]


def make_positions(config: SimConfig, seed: int):
    rng = random.Random(seed)
    return {
        node_id: (
            rng.uniform(0, config.area_size),
            rng.uniform(0, config.area_size),
        )
        for node_id in range(config.nb_nodes)
    }


def safe_lifetime(value, duration):
    """
    Si aucun seuil de mortalité n’est atteint, la valeur est None.
    On considère alors que la durée de vie est au moins égale à duration.
    """
    return duration if value is None else value


def run_one_simulation(
    config: SimConfig,
    protocol: ProtocolConfig,
    trace_file: str,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    positions = make_positions(config, seed)

    sim = Simulation(
        config=config,
        protocol=protocol,
        node_positions=positions,
        trace_file=trace_file,
        traffic_seed=seed,
    )

    sim.run()
    return sim.get_metrics()


def evaluate_candidate(config: SimConfig, protocol: ProtocolConfig):
    results = []

    for seed, trace_file in zip(SEEDS, TRACE_FILES):
        metrics = run_one_simulation(
            config=config,
            protocol=protocol,
            trace_file=trace_file,
            seed=seed,
        )
        results.append(metrics)

    delivery_ratios = [
        100.0 * r["msg_recv"] / max(1, r["messages_initiated"])
        for r in results
    ]

    ten_percent_deaths = [
        safe_lifetime(r["ten_percent_death"], config.duration)
        for r in results
    ]

    energies = [
        r["energy"]
        for r in results
    ]

    final_std_bats = [
        r["final_std_bat"]
        for r in results
    ]

    control_overheads = [
        (r["rreq_sent"] + r["rreq_forwarded"] + r["rrep_sent"])
        / max(1, r["messages_initiated"])
        for r in results
    ]

    return {
        "delivery_ratio": mean(delivery_ratios),
        "ten_percent_death": mean(ten_percent_deaths),
        "energy": mean(energies),
        "final_std_bat": mean(final_std_bats),
        "control_overhead": mean(control_overheads),
    }


class AodvOptimizationProblem(ElementwiseProblem):
    """
    Variables de décision x :

    x[0] = coeff_dist_weight, a
    x[1] = x_min
    x[2] = x_safe
    x[3] = seuil_coeff
    x[4] = max_duplicates, arrondi en entier
    x[5] = weight_seuil
    x[6] = p_short
    x[7] = p_long
    x[8] = p_bat
    x[9] = critical_bat_penalty
    x[10] = ttl_max, arrondi en entier

    Objectifs, tous exprimés en minimisation pour pymoo :

    f1 = - delivery_ratio
    f2 = - ten_percent_death
    f3 = energy
    f4 = final_std_bat
    f5 = control_overhead
    """

    def __init__(self):
        super().__init__(
            n_var=11,
            n_obj=5,
            n_constr=1,
            xl=np.array([
                0.0,    # a
                0.05,   # x_min
                0.60,   # x_safe
                0.03,   # seuil_coeff
                1.0,    # max_duplicates
                1.0,    # weight_seuil
                1.0,    # p_short
                1.0,    # p_long
                1.0,    # p_bat
                0.0,    # critical_bat_penalty
                4.0,    # ttl_max
            ]),
            xu=np.array([
                1.0,    # a
                0.40,   # x_min
                0.95,   # x_safe
                0.30,   # seuil_coeff
                5.0,    # max_duplicates
                2.5,    # weight_seuil
                4.0,    # p_short
                4.0,    # p_long
                4.0,    # p_bat
                5.0,    # critical_bat_penalty
                12.0,   # ttl_max
            ]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        a = float(x[0])
        b = 1.0 - a

        x_min = float(x[1])
        x_safe = float(x[2])

        max_duplicates = int(round(x[4]))
        ttl_max = int(round(x[10]))

        config = replace(
            BASE_CONF,
            coeff_dist_weight=a,
            coeff_bat_weight=b,
            x_min=x_min,
            x_safe=x_safe,
            seuil_coeff=float(x[3]),
            max_dist=BASE_CONF.max_dist,
            p_short=float(x[6]),
            p_long=float(x[7]),
            p_bat=float(x[8]),
            critical_bat_penalty=float(x[9]),
            ttl_max=ttl_max,
        )

        protocol = ProtocolConfig(
            reg_aodv=False,
            max_duplicates=max_duplicates,
            weight_seuil=float(x[5]),
        )

        metrics = evaluate_candidate(config, protocol)

        delivery_ratio = metrics["delivery_ratio"]
        ten_percent_death = metrics["ten_percent_death"]
        energy = metrics["energy"]
        final_std_bat = metrics["final_std_bat"]
        control_overhead = metrics["control_overhead"]

        # pymoo minimise tous les objectifs.
        out["F"] = np.array([
            -delivery_ratio,
            -ten_percent_death,
            energy,
            final_std_bat,
            control_overhead,
        ])

        # Contrainte technique :
        # x_min doit rester inférieur à x_safe.
        # pymoo considère g(x) <= 0 comme faisable.
        out["G"] = np.array([
            x_min - x_safe + 0.05
        ])


def main():
    problem = AodvOptimizationProblem()

    algorithm = NSGA2(
        pop_size=32,
        eliminate_duplicates=True,
    )

    # 10 générations de 32 individus = environ 320 évaluations.
    termination = get_termination("n_gen", 10)

    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=42,
        verbose=True,
        save_history=True,
    )

    X = result.X
    F = result.F

    rows = []

    for i in range(len(X)):
        x = X[i]
        f = F[i]

        rows.append({
            "solution": i,

            # Objectifs remis dans le sens naturel
            "delivery_ratio": -f[0],
            "ten_percent_death": -f[1],
            "energy": f[2],
            "final_std_bat": f[3],
            "control_overhead": f[4],

            # Paramètres optimisés
            "coeff_dist_weight": x[0],
            "coeff_bat_weight": 1.0 - x[0],
            "x_min": x[1],
            "x_safe": x[2],
            "seuil_coeff": x[3],
            "max_duplicates": int(round(x[4])),
            "weight_seuil": x[5],
            "p_short": x[6],
            "p_long": x[7],
            "p_bat": x[8],
            "critical_bat_penalty": x[9],
            "ttl_max": int(round(x[10])),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["delivery_ratio", "ten_percent_death", "energy"],
        ascending=[False, False, True],
    )

    df.to_csv(
        "pareto_pymoo_nsga2_aodv.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFront de Pareto exporté dans : pareto_pymoo_nsga2_aodv.csv")
    print(df.head(10))


if __name__ == "__main__":
    main()