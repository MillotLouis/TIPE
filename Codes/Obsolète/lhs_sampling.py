# lhs_sampling.py
import numpy as np
import pandas as pd

def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n + 1)
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        u = rng.random(n)
        s = edges[:-1] + u * (edges[1:] - edges[:-1])  # 1 point / strate
        rng.shuffle(s)                                  # permutation par dimension
        out[:, j] = s
    return out

def _stochastic_round(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    frac = x - np.floor(x)
    up = rng.random(x.shape) < frac
    return (np.floor(x) + up).astype(int)

def _map_to_range(u: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    t = spec["type"]
    if t == "float":
        lo, hi = spec["bounds"]
        if spec.get("log", False):
            lo_l, hi_l = np.log(lo), np.log(hi)
            return np.exp(lo_l + u * (hi_l - lo_l))
        return lo + u * (hi - lo)
    elif t == "int":
        lo, hi = spec["bounds"]
        cont = lo + u * (hi + 1 - lo)  # [lo, hi+1)
        return _stochastic_round(cont, rng)
    else:
        raise ValueError("Only 'float' or 'int' here (categoricals excluded).")

def generate_lhs(param_specs: dict, n: int, seed: int = 2025) -> pd.DataFrame:
    """
    Génère un plan LHS pour paramètres continus/entiers.
    'param_specs' : dict{name: {"type": "float"|"int", "bounds": (lo, hi), "log": bool?}}
    Retourne un DataFrame avec une colonne par paramètre + une colonne 'conso' assemblée si
    'conso_routing' et 'conso_data' existent.
    """
    rng = np.random.default_rng(seed)
    keys = list(param_specs.keys())
    d = len(keys)
    U = _lhs_unit(n, d, rng)
    data = {}
    for j, k in enumerate(keys):
        data[k] = _map_to_range(U[:, j], param_specs[k], rng)
    df = pd.DataFrame(data)

    # réassemblage 'conso' si applicable
    if "conso_routing" in df.columns and "conso_data" in df.columns:
        df["conso"] = list(zip(df["conso_routing"], df["conso_data"]))

    return df

# Exemple de bornes adaptées (ajuste au besoin)
DEFAULT_SPECS = {
    "nb_nodes":      {"type": "int",   "bounds": (30, 150)},
    "area_size":     {"type": "float", "bounds": (200.0, 1200.0)},
    "max_dist":      {"type": "float", "bounds": (20.0, 300.0)},
    "conso_routing": {"type": "float", "bounds": (0.01, 0.30), "log": True},
    "conso_data":    {"type": "float", "bounds": (0.02, 0.80), "log": True},
    "seuil":         {"type": "float", "bounds": (0.05, 0.95)},
    "coeff_dist":    {"type": "float", "bounds": (1e-4, 0.05), "log": True},
    "coeff_bat":     {"type": "float", "bounds": (0.2, 2.0)},
    "coeff_conso":   {"type": "float", "bounds": (1e-4, 0.2), "log": True},
    "ttl":           {"type": "int",   "bounds": (3, 25)},
    "init_bat":      {"type": "float", "bounds": (80.0, 250.0)},
}

if __name__ == "__main__":
    N = 10
    df = generate_lhs(DEFAULT_SPECS, n=N, seed=2025)
    cols = ["nb_nodes","area_size","max_dist","conso_routing","conso_data","conso",
            "seuil","coeff_dist","coeff_bat","coeff_conso","ttl","init_bat"]
    df = df[cols]
    df.to_csv("params_lhs_10.csv", index=False)
    print("Saved params_lhs_10.csv")
