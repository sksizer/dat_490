####Author: Jaime Kirk
####Created: 6/2/25
####Last Update: 6/15/25


import j_process
from IPython.display import clear_output
import time
import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd

#p_core_ids = list(range(0, 16))

# Set affinity for current process
#p = psutil.Process(os.getpid())
#p.cpu_affinity(p_core_ids)

from joblib import Parallel, delayed
from joblib import cpu_count



def resolve_cores(cores):
    available = cpu_count()
    if isinstance(cores, str):
        cores = cores.strip().lower()
        if cores == "50%":
            return available if available == 2 else max(1, available // 2)
        elif cores == "75%":
            return max(1, int(available * 0.75))
        elif cores in {"all", "max"}:
            return available
        elif cores == "n-1":
            return max(1, available - 1)
        else:
            raise ValueError(f"Unrecognized core spec: {cores}")
    elif isinstance(cores, int):
        return max(1, min(cores, available))
    else:
        raise TypeError("`cores` must be an int or string like '50%', 'all', etc.")


from joblib import Parallel, delayed

from joblib import Parallel, delayed
import numpy as np

def kmode_tune(train_set, features, s_cluster=2, n_cluster=256, n_trials=5, use_multiprocessing=True, cores="50%"):
    results = []

    print(f"Starting k-modes tuning from k={s_cluster} to {n_cluster - 1} on train_set only...")

    def run_single_trial(k):
        _, _, train_cost = j_process.run_kmodes_cluster(train_set, features, n_clusters=k, verbose=0)
        return train_cost

    def run_trials_for_k(k):
        print(f"Running k={k} ({n_trials} trials)...")
        if use_multiprocessing:
            train_costs = Parallel(n_jobs=cores, backend="loky")(
                delayed(run_single_trial)(k) for _ in range(n_trials)
            )
        else:
            train_costs = [run_single_trial(k) for _ in range(n_trials)]

        avg_train_cost = np.mean(train_costs)
        print(f"✓ k={k} | Avg Train Cost: {avg_train_cost:.2f}")
        return {
            "Clusters": k,
            "AvgTrainCost": avg_train_cost,
            "AllTrainCosts": train_costs
        }

    # Loop over values of k sequentially
    for k in range(s_cluster, n_cluster):
        results.append(run_trials_for_k(k))

    return results

def print_trial_status(i, accs, total=None, elapsed=None):
    clear_output(wait=True)
    print(f"Trial {i}/{total}" if total else f"Trial {i}")
    print(f"  CHRONIC:  {accs[0]:.4f}")
    print(f"  CARDIAC:  {accs[1]:.4f}")
    print(f"  PUL:      {accs[2]:.4f}")
    if elapsed is not None:
        print(f"  Elapsed Time: {elapsed:.1f}s")

def tflow_tune(train_set, features, s_cluster=2, n_cluster=64, n_trials=5):
    results = []

    print("Starting silhouette trials...")
    start_time = time.time()

    for k in range(s_cluster, n_cluster):
        sils = []

        for trial in range(1, n_trials + 1):
            tdf_out, kmodename, t_encoded = j_process.run_tf_clustering(train_set, features, n_clusters=k, verbose=0)
            silhouette = silhouette_score(t_encoded, tdf_out[kmodename])
            sils.append(silhouette)

            elapsed = time.time() - start_time
            clear_output(wait=True)
            print(f"k = {k} | Trial {trial}/{n_trials}")
            print(f"  Current Silhouette: {silhouette:.4f}")
            print(f"  Elapsed Time: {elapsed:.1f}s")

        avg_sil = np.mean(sils)
        clear_output(wait=True)
        print(f"k = {k} completed")
        print(f"  Avg Silhouette: {avg_sil:.4f}")
        print(f"  Elapsed Time: {time.time() - start_time:.1f}s")

        results.append({
            "Clusters": k,
            "AvgSilhouette": avg_sil,
            "AllSilhouettes": sils
        })

    return results

import matplotlib.pyplot as plt
import pandas as pd

def plot_kmode_elbow(df, train_col="AvgTrainCost",  iteration_col="Clusters"):
    """
    Plot the elbow curve from a tuning result DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Iteration', 'Train Cost', and optionally 'Val Cost'.
        train_col (str): Column name for training cost.
        val_col (str): Column name for validation cost.
        iteration_col (str): Column with cluster counts (e.g., 2 to N).
    """
    # Drop the baseline (Iteration 0) and any rows with NaNs
    df_plot = df[df[iteration_col] > 0].copy()
    df_plot = df_plot.dropna(subset=[train_col])

    plt.figure(figsize=(8, 5))
    plt.plot(df_plot[iteration_col], df_plot[train_col], marker='o', label='Train Cost')

\

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Cost (Mismatch Count)')
    plt.title('Elbow Curve for KModes Clustering')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Kmode_elbow.jpg", format='jpg', dpi=300)
    plt.show()
def analyze_silhouette_scores(results_df, top_n=5):
    """
    Analyzes silhouette scores from tflow_tune results passed as a DataFrame.

    Parameters:
        results_df (pd.DataFrame): DataFrame with at least columns:
            - 'Clusters' (int)
            - 'AvgSilhouette' (float)
        top_n (int): How many top scoring k values to return (default = 5)

    Returns:
        best_k (int): Cluster count with highest average silhouette score
        best_score (float): The highest silhouette score
        top_k_scores (list of tuples): Top-N (k, score) pairs sorted by score descending
    """
    ks = results_df["Clusters"].tolist()
    scores = results_df["AvgSilhouette"].tolist()

    # Plot silhouette vs. k
    plt.figure(figsize=(10, 5))
    plt.plot(ks, scores, marker='o')
    plt.title("Average Silhouette Score by Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.grid(True)
    plt.savefig("TFLOW_silhouette.jpg", format='jpg', dpi=300)
    plt.show()

    # Find best k
    best_index = scores.index(max(scores))
    best_k = ks[best_index]
    best_score = scores[best_index]

    # Top-N k values sorted by score
    top_k_scores = sorted(zip(ks, scores), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"Best k: {best_k} with silhouette score = {best_score:.4f}")
    print(f"Top {top_n} k values:")
    for k, s in top_k_scores:
        print(f"  k = {k}, score = {s:.4f}")

    return best_k, best_score, top_k_scores