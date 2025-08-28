# plot_simple.py
# Objectif : graphiques simples, lisibles, basés sur ton results.csv
# - Barres : delivery ratio moyen et énergie moyenne par protocole (avec barres d'erreur)
# - Nuage énergie vs delivery + médiane glissante
# - Effet d'un paramètre choisi : moyennes par quartiles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============== Utilitaires sûrs =================

def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Colonnes manquantes. Aucune parmi: {candidates}")
    return None

def compute_delivery_ratio(df: pd.DataFrame) -> np.ndarray:
    # Dans ton CSV : msg_recv, messages_initiated existent
    recv_col = pick_col(df, ["msg_recv", "messages_received"])
    sent_col = pick_col(df, ["messages_initiated", "msg_sent"], required=False)
    if sent_col is None:
        raise ValueError("Il faut 'messages_initiated' (ou à défaut 'msg_sent') dans le CSV.")
    recv = df[recv_col].astype(float).values
    den  = np.maximum(df[sent_col].astype(float).values, 1.0)
    return recv / den

def energy_col_name(df: pd.DataFrame) -> str:
    return pick_col(df, ["energy", "total_energy", "energy_consumed"])

def mean_and_sem(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    n = np.sum(~np.isnan(x))
    if n <= 1:
        return m, 0.0
    s = np.nanstd(x, ddof=1) / np.sqrt(n)  # erreur standard
    return m, s

def running_median(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    # Courbe médiane glissante pour guider l’œil
    # Binner x en quantiles pour avoir ~autant de points par bin
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < max(10, nbins):
        return np.array([]), np.array([])

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x, qs)
    xm, ym = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (x >= a) & (x <= b) if b == edges[-1] else (x >= a) & (x < b)
        if np.any(mask):
            xm.append(np.median(x[mask]))
            ym.append(np.median(y[mask]))
    return np.array(xm), np.array(ym)

def approx_pareto_front(energy: np.ndarray, delivery: np.ndarray, k: int = 50):
    # Renvoie jusqu'à k points non dominés approximatifs (échantillon + filtre)
    # Minimiser energy, maximiser delivery
    idx = np.arange(len(energy))
    # échantillonner si trop de points
    if len(idx) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=5000, replace=False)
    e, d, id_sel = energy[idx], delivery[idx], idx
    # tri par énergie croissante
    order = np.argsort(e)
    e, d, id_sel = e[order], d[order], id_sel[order]
    # filtre non dominé (croissance de d)
    keep = []
    best_d = -np.inf
    for i in range(len(e)):
        if d[i] > best_d:
            keep.append(id_sel[i])
            best_d = d[i]
        if len(keep) >= k:
            break
    return np.array(keep, dtype=int)

# =============== Figures =================

def bars_by_protocol(df: pd.DataFrame):
    """Deux figures en barres : delivery moyen et énergie moyenne, par protocole."""
    if "protocol" not in df.columns:
        print("⚠ Pas de colonne 'protocol' : graphes par protocole sautés.")
        return

    df = df.copy()
    df["delivery_ratio"] = compute_delivery_ratio(df)
    e_col = energy_col_name(df)

    # 1) Delivery ratio
    labels, means, sems = [], [], []
    for name, g in df.groupby("protocol"):
        m, s = mean_and_sem(g["delivery_ratio"].values)
        labels.append(name); means.append(m); sems.append(s)

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Taux de livraison (moyenne ± erreur standard)")
    plt.title("Comparaison protocolaire — Delivery ratio")
    plt.tight_layout()
    plt.show()

    # 2) Énergie
    labels, means, sems = [], [], []
    for name, g in df.groupby("protocol"):
        m, s = mean_and_sem(g[e_col].values)
        labels.append(name); means.append(m); sems.append(s)

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(f"{e_col} (moyenne ± erreur standard)")
    plt.title("Comparaison protocolaire — Énergie consommée")
    plt.tight_layout()
    plt.show()

def scatter_energy_vs_delivery(df: pd.DataFrame, show_pareto=True):
    """Nuage énergie vs delivery, médiane glissante, front Pareto optionnel."""
    df = df.copy()
    e_col = energy_col_name(df)
    df["delivery_ratio"] = compute_delivery_ratio(df)

    plt.figure()
    if "protocol" in df.columns:
        for name, g in df.groupby("protocol"):
            plt.scatter(g[e_col].values, g["delivery_ratio"].values, s=10, alpha=0.5, label=name)
        plt.legend()
    else:
        plt.scatter(df[e_col].values, df["delivery_ratio"].values, s=10, alpha=0.5)

    # médiane glissante (toutes données)
    xm, ym = running_median(df[e_col].values, df["delivery_ratio"].values, nbins=25)
    if len(xm):
        plt.plot(xm, ym, linewidth=2)

    # front Pareto approximatif
    if show_pareto:
        ids = approx_pareto_front(df[e_col].values, df["delivery_ratio"].values, k=60)
        if len(ids):
            plt.plot(df.iloc[ids][e_col].values, df.iloc[ids]["delivery_ratio"].values, linewidth=2)

    plt.xlabel(e_col)
    plt.ylabel("Taux de livraison")
    plt.title("Compromis énergie ↔ livraison")
    plt.tight_layout()
    plt.show()

def quartile_effect(df: pd.DataFrame, param_name: str):
    """
    Effet moyen d'un paramètre : on découpe param_name en 4 quartiles,
    puis on trace :
      - moyenne du delivery ratio par quartile
      - moyenne de l'énergie par quartile
    Deux figures simples, lisibles.
    """
    if param_name not in df.columns:
        print(f"⚠ Paramètre '{param_name}' introuvable dans results.csv.")
        return

    df = df.copy()
    e_col = energy_col_name(df)
    df["delivery_ratio"] = compute_delivery_ratio(df)

    # quartiles
    q = np.quantile(df[param_name].values, [0, 0.25, 0.5, 0.75, 1.0])
    labels = [f"[{q[i]:.3g},{q[i+1]:.3g}{']' if i==3 else ')'}" for i in range(4)]
    idxs = []
    for i in range(4):
        if i < 3:
            m = (df[param_name] >= q[i]) & (df[param_name] < q[i+1])
        else:
            m = (df[param_name] >= q[i]) & (df[param_name] <= q[i+1])
        idxs.append(np.where(m)[0])

    # Figure 1 : delivery ratio
    means, sems = [], []
    for idv in idxs:
        m, s = mean_and_sem(df.iloc[idv]["delivery_ratio"].values)
        means.append(m); sems.append(s)
    x = np.arange(4)
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Taux de livraison (moyenne ± erreur standard)")
    plt.title(f"Effet de '{param_name}' sur le delivery (quartiles)")
    plt.tight_layout()
    plt.show()

    # Figure 2 : énergie
    means, sems = [], []
    for idv in idxs:
        m, s = mean_and_sem(df.iloc[idv][e_col].values)
        means.append(m); sems.append(s)
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(f"{e_col} (moyenne ± erreur standard)")
    plt.title(f"Effet de '{param_name}' sur l'énergie (quartiles)")
    plt.tight_layout()
    plt.show()

# =============== Main =================

if __name__ == "__main__":
    in_path = Path("results.csv")
    df = pd.read_csv(in_path)

    # 1) Résumés par protocole (barres)
    bars_by_protocol(df)

    # 2) Compromis énergie ↔ livraison (scatter + médiane + Pareto)
    scatter_energy_vs_delivery(df, show_pareto=True)

    # 3) Effet d'un paramètre (exemples : 'ttl', 'max_dist', 'coeff_conso', 'nb_nodes')
    #   -> change PARAM ci-dessous selon ce que tu veux analyser
    PARAM = "ttl"
    quartile_effect(df, PARAM)
