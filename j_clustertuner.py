import j_process

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


def kmode_tune(train_set, val_set, features, use_multiprocessing=True, cores=,n_cluster=64):
    results = []
    der_var = ["ALL_CHRONIC", "ALL_CARDIAC", "ALL_PUL"]

    print("Starting Baseline...")
    base_score_all = j_process.run_logistic_model(train_set, val_set, features, der_var[0])
    base_score_car = j_process.run_logistic_model(train_set, val_set, features, der_var[1])
    base_score_pul = j_process.run_logistic_model(train_set, val_set, features, der_var[2])
    results.append({
        "Iteration": 0,
        "Performance": [base_score_all, base_score_car, base_score_pul],
        "Train Clusters": [],
        "Val Clusters": []
    })

    def run_trial(i):
        print(f"Starting trial {i}")
        tdf_out, kmodename = j_process.run_kmodes_cluster(train_set, features, n_clusters=i, verbose=0)
        vdf_out, kmodename2 = j_process.run_kmodes_cluster(val_set, features, n_clusters=i, verbose=0)

        tfeatures = features.copy() + [kmodename]
        all_acc_score = j_process.run_logistic_model(tdf_out, vdf_out, tfeatures, der_var[0])
        car_acc_score = j_process.run_logistic_model(tdf_out, vdf_out, tfeatures, der_var[1])
        pul_acc_score = j_process.run_logistic_model(tdf_out, vdf_out, tfeatures, der_var[2])

        print(f"Finished trial {i}")
        return {
            "Iteration": i,
            "Performance": [all_acc_score, car_acc_score, pul_acc_score],
            "Train Clusters": tdf_out[kmodename].tolist(),
            "Val Clusters": vdf_out[kmodename2].tolist()
        }

    print("Beginning Trials")
    if use_multiprocessing:
        print(f"Running in parallel with {cores} cores.")
        parallel_results = Parallel(n_jobs=cores, backend="loky")(
            delayed(run_trial)(i) for i in range(2, n_cluster)
        )
        results.extend(parallel_results)
    else:
        print("Running sequentially.")
        for i in range(2, n_cluster):
            results.append(run_trial(i))

    return results


def tflow_tune(train_set, val_set, features, n_cluster=64):
    results=[]
    der_var=["ALL_CHRONIC","ALL_CARDIAC","ALL_PUL"]

    print("Starting Baseline...")
    print("Starting Baseline...")
    base_score_all = j_process.run_logistic_model(train_set,val_set,features,der_var[0])
    base_score_car = j_process.run_logistic_model(train_set,val_set,features,der_var[1])
    base_score_pul = j_process.run_logistic_model(train_set,val_set,features,der_var[2])
    results.append({
        "Iteration": 0,
        "Performance": [base_score_all,base_score_car,base_score_pul],
        "Train Clusters" : [],
        "Val Clusters" : []
    })
    
    print("Beginning Trials")
    for i in range(2,n_cluster):
        print("Trial")
        print(i)
        print("Clustering training set...")
        tdf_out, kmodename = j_process.run_tf_clustering(train_set,features,n_clusters=i,verbose=0)
        print("Clustering valadation set...")
        vdf_out, kmodename2 = j_process.run_tf_clustering(val_set,features,n_clusters=i,verbose=0)
        tfeatures = features.copy() + [kmodename]
        all_acc_score = j_process.run_logistic_model(tdf_out,vdf_out,tfeatures,der_var[0])
        car_acc_score = j_process.run_logistic_model(tdf_out,vdf_out,tfeatures,der_var[1])
        pul_acc_score = j_process.run_logistic_model(tdf_out,vdf_out,tfeatures,der_var[2])
        
        results.append({
            "Iteration": i,
            "Performance": [all_acc_score, car_acc_score,pul_acc_score],
            "Train Clusters": tdf_out[kmodename].tolist(),
            "Val Clusters": vdf_out[kmodename2].tolist()
        })
    

    return results