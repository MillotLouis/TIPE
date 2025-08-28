# plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Helpers robustes ----------

def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """
    Retourne le premier nom de colonne présent dans df parmi 'candidates'.
    Si 'required' et aucun trouvé, lève une erreur explicite.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Colonnes manquantes. Aucune parmi: {candidates}")
    return None

def compute_delivery_ratio(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule delivery_ratio = msg_recv / messages_initiated (par défaut),
    sinon msg_recv / msg_sent si 'messages_initiated' n'existe pas.
    """
    recv_col = pick_col(df, ["msg_recv", "messages_received"])
    # on privilégie 'messages_initiated' (présent dans ton CSV),
    # sinon on retombe sur 'msg_sent'.
    sent_base = pick_col(df, ["messages_initiated", "msg_sent"], required=False)
    if sent_base is None:
        raise ValueError("Impossible de trouver 'messages_initiated' ou 'msg_sent' dans le CSV.")
    recv = df[recv_col].astype(float).values
    den  = np.maximum(df[sent_base].astype(float).values, 1.0)  # évite /0
    return recv / den

def compute_energy_per_delivered(df: pd.DataFrame, energy_col: str) -> np.ndarray:
    """
    Énergie par message délivré: energy / max(msg_recv,1).
    Utile pour mesurer l'efficience énergétique.
    """
    recv_col = pick_col(df, ["msg_recv", "messages_received"])
    delivered = np.maximum(df[recv_col].astype(float).values, 1.0)
    energy = df[energy_col].astype(float).values
    return energy / delivered

# ---------- Figures ----------

def fig_pareto_energy_vs_delivery(df: pd.DataFrame):
    """
    Nuage Pareto: énergie vs delivery_ratio, séparé par 'protocol'.
    """
    energy_col = pick_col(df, ["energy", "total_energy", "energy_consumed"])
    df = df.copy()
    df["delivery_ratio"] = compute_delivery_ratio(df)

    plt.figure()
    if "protocol" in df.columns:
        for name, g in df.groupby("protocol"):
            plt.scatter(g[energy_col].values, g["delivery_ratio"].values, s=10, alpha=0.5, label=name)
        plt.legend()
    else:
        plt.scatter(df[energy_col].values, df["delivery_ratio"].values, s=10, alpha=0.5)

    plt.xlabel(energy_col)
    plt.ylabel("delivery_ratio")
    plt.title("Énergie vs Taux de livraison (Pareto)")
    plt.tight_layout()
    plt.show()

def fig_parallel_coordinates(df: pd.DataFrame):
    """
    Coordonnées parallèles sur quelques paramètres + une métrique (energy).
    Choisis ici des colonnes présentes dans ton CSV.
    """
    energy_col = pick_col(df, ["energy", "total_energy", "energy_consumed"])
    param_cols = [
        # paramètres LHS présents dans ton CSV
        "nb_nodes", "max_dist", "coeff_dist", "coeff_conso", "ttl"
    ]
    # garde uniquement les colonnes dispo
    param_cols = [c for c in param_cols if c in df.columns]
    cols = param_cols + [energy_col]
    sub = df[cols].copy()

    # échantillonnage pour la lisibilité
    if len(sub) > 400:
        sub = sub.sample(400, random_state=0)

    # normalisation min-max
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-12)

    x = np.arange(len(cols))
    plt.figure()
    for _, row in sub_norm.iterrows():
        plt.plot(x, row.values, alpha=0.15)
    plt.xticks(x, cols, rotation=30, ha="right")
    plt.title("Coordonnées parallèles (paramètres normalisés + énergie)")
    plt.tight_layout()
    plt.show()

def fig_hist_params(df: pd.DataFrame):
    """
    Histogrammes des paramètres (contrôle du plan LHS).
    Adapte la liste aux colonnes que tu veux visualiser.
    """
    hist_cols = [
        "nb_nodes","area_size","max_dist","conso_routing","conso_data",
        "seuil","coeff_dist","coeff_bat","coeff_conso","ttl"  # 'init_bat' si tu l'as varié
    ]
    hist_cols = [c for c in hist_cols if c in df.columns]
    for c in hist_cols:
        plt.figure()
        plt.hist(df[c].values, bins=40)
        plt.title(f"Distribution de {c}")
        plt.xlabel(c); plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.show()

def fig_energy_per_delivered(df: pd.DataFrame):
    """
    Scatter: énergie par message délivré vs delivery_ratio (lecture d’efficience).
    """
    energy_col = pick_col(df, ["energy", "total_energy", "energy_consumed"])
    df = df.copy()
    df["delivery_ratio"] = compute_delivery_ratio(df)
    df["energy_per_deliv"] = compute_energy_per_delivered(df, energy_col)

    plt.figure()
    if "protocol" in df.columns:
        for name, g in df.groupby("protocol"):
            plt.scatter(g["delivery_ratio"].values, g["energy_per_deliv"].values, s=10, alpha=0.5, label=name)
        plt.legend()
    else:
        plt.scatter(df["delivery_ratio"].values, df["energy_per_deliv"].values, s=10, alpha=0.5)

    plt.xlabel("delivery_ratio")
    plt.ylabel("energy_per_deliv")
    plt.title("Énergie par message délivré vs Taux de livraison")
    plt.tight_layout()
    plt.show()

# ---------- Main ----------

if __name__ == "__main__":
    # Nom du fichier résultats (adapte si besoin)
    in_path = Path("results.csv")
    df = pd.read_csv(in_path)

    # Figures
    fig_pareto_energy_vs_delivery(df)
    fig_parallel_coordinates(df)
    # fig_hist_params(df)
    fig_energy_per_delivered(df)
