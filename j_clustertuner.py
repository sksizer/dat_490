import j_process
def kmode_tune(train_set, val_set, features,target):
    results=[]
    print("Starting Baseline...")
    base_score = run_logistic_model(train_set,val_set,features,target)
    results.append({
        "Iteration": 0,
        "Performance": base_score,
        "Train Clusters" : [],
        "Val Clusters" : []
    })
    i=1
    print("Beginning Trials")
    while i != 65:
        print("Trial")
        print(i)
        tdf_out, kmodename = run_kmodes_cluster(train_set,features,n_clusters=i)
        vdf_out, kmodename2 = run_kmodes_cluster(val_set,features,n_clusters=i)
        tfeatures = features.copy() + [kmodename]
        acc_score = run_logistic_model(tdf_out,vdf_out,tfeatures,target)
        results.append({
            "Iteration": i,
            "Performance": acc_score,
            "Train Clusters": tdf_out[kmodename],
            "Val Clusters": vdf_out[kmodename2]
        })
        i = i + 1

    return results
        
    