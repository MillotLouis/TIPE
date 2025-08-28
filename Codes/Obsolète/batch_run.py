# batch_run.py
import numpy as np
import pandas as pd
from simulation import Simulation

def make_positions(nb_nodes: int, area_size: float, rng: np.random.Generator) -> list[tuple[float,float]]:
    # positions uniformes dans un carré [0, area_size]²
    xs = rng.random(nb_nodes) * area_size
    ys = rng.random(nb_nodes) * area_size
    return list(zip(xs, ys))

def run_one_config(row, rng_seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(rng_seed)
    nb_nodes   = int(row.nb_nodes)
    area_size  = float(row.area_size)
    max_dist   = float(row.max_dist)
    conso      = tuple(row.conso) if isinstance(row.conso, (list, tuple)) else eval(row.conso)
    seuil      = float(row.seuil)
    coeff_dist = float(row.coeff_dist)
    coeff_bat  = float(row.coeff_bat)
    coeff_conso= float(row.coeff_conso)
    ttl        = int(row.ttl)
    init_bat   = float(row.init_bat)

    positions = make_positions(nb_nodes, area_size, rng)

    results = []
    for reg in (True, False):
        sim = Simulation(
            nb_nodes=nb_nodes,
            area_size=area_size,
            max_dist=max_dist,
            conso=conso,
            seuil=seuil,
            coeff_dist=coeff_dist,
            coeff_bat=coeff_bat,
            coeff_conso=coeff_conso,
            ttl=ttl,
            reg_aodv=reg,
            init_bat=init_bat,
            node_positions=positions
        )
        sim.run()
        metrics = sim.get_metrics()  # suppose que ta classe Simulation a déjà cette méthode
        metrics["protocol"] = "AODV_reg" if reg else "AODV_mod"
        results.append(metrics)
    return results

def run_batch(csv_path: str, out_csv: str, seed: int = 2025, max_rows: int | None = None):
    df = pd.read_csv(csv_path, converters={"conso": eval})
    if max_rows is not None:
        df = df.head(max_rows).copy()

    all_rows = []
    for i, row in df.iterrows():
        pair = run_one_config(row, rng_seed=seed + i)
        # ajoute les paramètres bruts à chaque ligne de métriques
        for r in pair:
            r.update(row.to_dict())
        all_rows.extend(pair)
    out = pd.DataFrame(all_rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(out)} rows")

if __name__ == "__main__":
    run_batch("params_lhs_10.csv", "results.csv", seed=2025, max_rows=None)
