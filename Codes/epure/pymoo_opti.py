from __future__ import annotations

import os
from dataclasses import replace
from multiprocessing import Pool, cpu_count
from time import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from simulation_opti import (
    BonnMotionConfig,
    ProtocolConfig,
    SimConfig,
    generate_bonnmotion_traces,
    run_comparison_simulations,
)


class AodvOptiProblem(Problem):
    """NSGA-II problem:
    minimize [-ten_percent_death, -delivery_ratio, final_std_bat]
    """

    def __init__(self, base_conf: SimConfig, trace_files: list[str], nb_runs: int, seed_base: int, n_processes: int):
        self.base_conf = base_conf
        self.trace_files = trace_files
        self.nb_runs = nb_runs
        self.seed_base = seed_base
        self.n_processes = n_processes

        xl = np.array([0.10, 0.03, 0.20, 0.05, 0.60, 1.0, 1.00], dtype=float)
        xu = np.array([0.90, 0.20, 4.00, 0.35, 0.95, 3.0, 1.40], dtype=float)

        super().__init__(
            n_var=7,
            n_obj=3,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
        )

    @staticmethod
    def _decode(x):
        coeff_dist_weight = float(x[0])
        coeff_bat_weight = 1.0 - coeff_dist_weight
        seuil_coeff = float(x[1])
        penalite_seuil = float(x[2])
        d_min = float(x[3])
        d_max = float(x[4])
        max_duplicates = int(round(x[5]))
        weight_seuil = float(x[6])
        return {
            "coeff_dist_weight": coeff_dist_weight,
            "coeff_bat_weight": coeff_bat_weight,
            "seuil_coeff": seuil_coeff,
            "penalite_seuil": penalite_seuil,
            "d_min": d_min,
            "d_max": d_max,
            "max_duplicates": max_duplicates,
            "weight_seuil": weight_seuil,
        }

    def _evaluate_one(self, x: np.ndarray):
        p = self._decode(x)
        config = replace(
            self.base_conf,
            coeff_dist_weight=p["coeff_dist_weight"],
            coeff_bat_weight=p["coeff_bat_weight"],
            seuil_coeff=p["seuil_coeff"],
            penalite_seuil=p["penalite_seuil"],
            d_min=p["d_min"],
            d_max=p["d_max"],
        )
        protocol = ProtocolConfig(reg_aodv=False, max_duplicates=p["max_duplicates"], weight_seuil=p["weight_seuil"])

        res = run_comparison_simulations(
            config=config,
            nb_runs=self.nb_runs,
            seed_base=self.seed_base,
            trace_files=self.trace_files,
            protocols=[protocol],
            use_parallel_runs=True,
            n_processes=self.n_processes,
        )

        mod_avg = res["mod_avg"][0]
        ten_percent_death = mod_avg.get("ten_percent_death")
        delivery_ratio = 100.0 * mod_avg.get("msg_recv", 0) / max(1.0, mod_avg.get("messages_initiated", 1.0))
        final_std_bat = mod_avg.get("final_std_bat")

        if ten_percent_death is None:
            ten_percent_death = config.duration
        if final_std_bat is None:
            final_std_bat = 1e6

        f = np.array([-ten_percent_death, -delivery_ratio, final_std_bat], dtype=float)
        g = np.array([p["d_min"] - p["d_max"] + 1e-6], dtype=float)

        return f, g

    def _evaluate(self, X, out, *args, **kwargs):
        tasks = [np.array(row, dtype=float) for row in X]
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(self._evaluate_one, tasks)

        out["F"] = np.array([r[0] for r in results], dtype=float)
        out["G"] = np.array([r[1] for r in results], dtype=float)


def run_nsga2(
    sim_conf: SimConfig,
    bm_conf: BonnMotionConfig,
    nb_runs: int = 8,
    seed_base: int = 424242,
    pop_size: int = 48,
    n_gen: int = 30,
    n_processes: int | None = None,
):
    if n_processes is None:
        n_processes = min(max(1, cpu_count() - 1), 11)

    os.makedirs(bm_conf.out_dir, exist_ok=True)
    print("Generating BonnMotion traces ONCE...")
    trace_files = generate_bonnmotion_traces(sim_conf, bm_conf, nb_runs)

    problem = AodvOptiProblem(
        base_conf=sim_conf,
        trace_files=trace_files,
        nb_runs=nb_runs,
        seed_base=seed_base,
        n_processes=n_processes,
    )

    algo = NSGA2(pop_size=pop_size)
    term = get_termination("n_gen", n_gen)

    t0 = time()
    res = minimize(problem, algo, termination=term, seed=seed_base, verbose=True)
    dt = time() - t0

    print(f"Optimization done in {dt:.1f}s")
    print("Pareto X:")
    print(res.X)
    print("Pareto F:")
    print(res.F)

    return res


if __name__ == "__main__":
    sim_conf = SimConfig(
        nb_nodes=40,
        area_size=800,
        max_dist=250,
        init_bat=100,
        conso=(0.00164, 0.0082, 10),
        dt=0.25,
        ttl_max=7,
        seuil_coeff=0.075,
        coeff_dist_weight=0.6,
        coeff_bat_weight=0.4,
        duration=600,
        d_min=0.15,
        d_max=0.80,
        penalite_seuil=2.0,
    )

    bm_conf = BonnMotionConfig(
        bm_exe="C:/Users/millo/Documents/bonnmotion-3.0.1/bin/bm.bat",
        out_dir="C:/Users/millo/Documents/GitHub/TIPE/bm_files/",
        vmin=10,
        vmax=10,
        pause=5,
    )

    run_nsga2(
        sim_conf=sim_conf,
        bm_conf=bm_conf,
        nb_runs=8,
        seed_base=424242,
        pop_size=48,
        n_gen=30,
        n_processes=11,
    )
