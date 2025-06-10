import j_process
def kmode_tune(train_set, val_set, features,target):
    results=[]
    print("Starting Baseline...")
    base_score = j_process.run_logistic_model(train_set,val_set,features,target)
    results.append({
        "Iteration": 0,
        "Performance": base_score,
        "Train Clusters" : [],
        "Val Clusters" : []
    })
    i=2
    print("Beginning Trials")
    while i != 65:
        print("Trial")
        print(i)
        print("Clustering training set...")
        tdf_out, kmodename = j_process.run_kmodes_cluster(train_set,features,n_clusters=i,verbose=0)
        print("Clustering valadation set...")
        vdf_out, kmodename2 = j_process.run_kmodes_cluster(val_set,features,n_clusters=i,verbose=0)
        tfeatures = features.copy() + [kmodename]
        acc_score = j_process.run_logistic_model(tdf_out,vdf_out,tfeatures,target)
        results.append({
            "Iteration": i,
            "Performance": acc_score,
            "Train Clusters": tdf_out[kmodename],
            "Val Clusters": vdf_out[kmodename2]
        })
        i = i + 1

    return results
        
    