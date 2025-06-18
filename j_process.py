######Author: Jaime Kirk
######Created: 6/2/25
######Last updated:6/17/25
from IPython.display import clear_output
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from IPython.display import display, Markdown
#export TF_ENABLE_ONEDNN_OPTS=1
__docstrings__ = {}
import time

def help(command=None):
    if command is None:
        print("Available functions:")
        for cmd in __docstrings__:
            print(f"  - {cmd}")
        print("\nUse `j_process.help('function_name')` to get detailed info.")
    elif command in __docstrings__:
        print(__docstrings__[command])
    else:
        print(f"No help found for '{command}'")


# KModes Clustering
def run_kmodes_cluster(df, feature_cols, n_clusters=5, init_method='Huang',
                       n_init=5, verbose=0, cluster_col_name=None):
    """
    Run k-modes clustering on categorical features.

    Args:
        df (DataFrame): Input data.
        feature_cols (list): Columns to cluster.
        n_clusters (int): Number of clusters.
        init_method (str): 'Huang' or 'Cao'.
        n_init (int): Number of initializations.
        verbose (int): Verbosity.
        cluster_col_name (str): Custom name for cluster column.

    Returns:
        DataFrame with cluster column added,
        name of the new cluster column,
        and cost (sum of dissimilarities).
    """
    #print(f"Clustering on {len(feature_cols)} features: {feature_cols}") just needed for debugging
    X_cluster = df[feature_cols].astype(str)

    km = KModes(n_clusters=n_clusters, init=init_method, n_init=n_init, verbose=verbose)
    clusters = km.fit_predict(X_cluster)

    df = df.copy()
    if cluster_col_name is None:
        init_code = 'h' if init_method.lower() == 'huang' else 'c'
        cluster_col_name = f'kmode_n{n_clusters}_i{n_init}_{init_code}'

    df[cluster_col_name] = clusters
    return df, cluster_col_name, km.cost_
__docstrings__['run_kmodes_cluster'] = run_kmodes_cluster.__doc__


# TF+KMeans Clustering
def run_tf_clustering(df, feature_cols, n_clusters=5, latent_dim=8, cluster_col_name=None,
                      epochs=50, batch_size=512, verbose=0):
    """
    Run TensorFlow autoencoder + KMeans clustering.

    Args:
        df (DataFrame): Input data.
        feature_cols (list): Columns to use.
        n_clusters (int): Number of clusters.
        latent_dim (int): Bottleneck dimension.
        cluster_col_name (str): Optional cluster column name.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        verbose (int): Verbosity level.

    Returns:
        DataFrame with cluster column added.
    """
    X_raw = df[feature_cols].astype(str).fillna("Missing")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_raw)
    ### build and run dimensionality reduction
    input_dim = X_encoded.shape[1]
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(64, activation='relu')(inputs)
    bottleneck = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    ####Kmeans on reduced dimensions
    autoencoder = tf.keras.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_encoded, X_encoded, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #####Start fitting cluster groupings
    encoder_model = tf.keras.Model(inputs, bottleneck)
    latent_features = encoder_model.predict(X_encoded,verbose=verbose)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features)
    ####Automatically names column based on parameters (if not specified)
    if cluster_col_name is None:
        cluster_col_name = f"tf_n{n_clusters}_d{latent_dim}_e{epochs}"
    #### add custer column
    df = df.copy()
    df[cluster_col_name] = cluster_labels
    ####returns cluster column name for looping
    return df,cluster_col_name

__docstrings__['run_tf_clustering'] = run_tf_clustering.__doc__


# Cluster Overlap Plot
## This plot allows comparison between cluster outputs of different methods.
### A heatmap comparing quantities of observations in each cluster
## Intended to show the difference between tflow clustering and KMODE
def plot_cluster_overlap(df, cluster_col1, cluster_col2, title="Cluster Overlap",
                         xlabel=None, ylabel=None, cmap='Blues',
                         save_path=None, dpi=300, show_plot=True):
    """
    Plot heatmap of overlap between two cluster columns.
    
    Args:
        df: DataFrame with cluster columns.
        cluster_col1: First cluster column.
        cluster_col2: Second cluster column.
        title: Plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        cmap: Color map.
        save_path: If set, saves image.
        dpi: Image resolution.
        show_plot: Show or suppress plot.
    """
    ct = pd.crosstab(df[cluster_col1], df[cluster_col2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else cluster_col2)
    plt.ylabel(ylabel if ylabel else cluster_col1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='jpg', dpi=dpi, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
### used for help() function
__docstrings__['plot_cluster_overlap'] = plot_cluster_overlap.__doc__


# Feature Distribution Comparison
#This shows the distribution of responses by cluster. 
#Used to show the difference between kmode and tflow clustering***
#*** only gives a decent output if the same "K" value is used***
# *Note* EVERY kmode and tflow run is slightly different, as they use a random starting point
def compare_feature_distributions(df1, df2, features, label1='Group A', label2='Group B',
                                  normalize=True, save_prefix=None, show_plot=True, figsize=(8, 4)):
    """
    Compare distribution of features between two groups.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        features: Feature list to compare.
        label1: Label for df1.
        label2: Label for df2.
        normalize: Use proportions instead of counts.
        save_prefix: If set, saves plots.
        show_plot: Whether to display plots.
        figsize: Size of plot.
    """
    for feature in features:
        vc1 = df1[feature].value_counts(normalize=normalize).sort_index()
        vc2 = df2[feature].value_counts(normalize=normalize).sort_index()
        compare_df = pd.DataFrame({label1: vc1, label2: vc2}).fillna(0)
        fig, ax = plt.subplots(figsize=figsize)
        compare_df.plot(kind='bar', ax=ax, width=0.8)
        plt.title(f'{feature} Distribution: {label1} vs {label2}')
        plt.ylabel('Proportion' if normalize else 'Count')
        plt.xlabel(feature)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_prefix:
            filename = f"{save_prefix}_{feature.replace(' ', '_').lower()}.jpg"
            plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

__docstrings__['compare_feature_distributions'] = compare_feature_distributions.__doc__


# Binary feature distribution by cluster
def plot_yes_no_by_cluster(df, cluster_col, binary_features, yes_value='Yes'):
    """
    Plot Yes/No distributions for binary features by cluster.

    Args:
        df: Input DataFrame.
        cluster_col: Cluster column to group by.
        binary_features: List of binary categorical columns.
        yes_value: Value to count as 'yes' (default: 'Yes').
    """
    for feature in binary_features:
        ctab = pd.crosstab(df[cluster_col], df[feature], normalize='index').fillna(0)
        if 'Yes' in ctab.columns and 'No' in ctab.columns:
            ctab = ctab[['Yes', 'No']]
        elif yes_value in ctab.columns:
            others = [col for col in ctab.columns if col != yes_value]
            ctab = ctab[[yes_value] + others]
        ctab.plot(kind='bar', stacked=True, figsize=(7, 4))
        plt.title(f'Distribution of {feature} by {cluster_col}')
        plt.ylabel('Proportion')
        plt.xlabel('Cluster')
        plt.xticks(rotation=0)
        plt.legend(title=feature)
        plt.tight_layout()
        plt.show()

__docstrings__['plot_yes_no_by_cluster'] = plot_yes_no_by_cluster.__doc__
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
### Runs the logistic model
def run_logistic_model(
    df_train,
    df_val,
    features,
    target,
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    encoder_drop=None,
    encoder_dtype=None,
    encoder_min_frequency=None,
    average='binary',
    class_weight='balanced',
    plot_importance=False,
    importance_filename=None
):
    if not plot_importance:
        
        clear_output(wait=True)
    print(f"Running logistic regression for target: {target}:")

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        drop=encoder_drop,
        dtype=encoder_dtype,
        min_frequency=encoder_min_frequency
    )

    X_train = encoder.fit_transform(df_train[features].astype(str))
    X_val = encoder.transform(df_val[features].astype(str))

    y_train = df_train[target]
    y_val = df_val[target]

    if y_train.dtype == 'object' or isinstance(y_train.iloc[0], str):
        unique_vals = sorted(y_train.dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Target column must be binary. Found: {unique_vals}")
        target_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y_train = y_train.map(target_map)
        y_val = y_val.map(target_map)
        print(f"Target mapped as: {target_map}")
    else:
        target_map = None 

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average=average, pos_label=1, zero_division=0),
        "recall": recall_score(y_val, y_pred, average=average, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, average=average, pos_label=1, zero_division=0)
    }

    print(f"Validation Scores for {target} -> Accuracy: {results['accuracy']:.4f}, "
          f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, "
          f"F1: {results['f1_score']:.4f}")

    if plot_importance:
        encoded_feature_names = encoder.get_feature_names_out()
        original_feature_names = encoder.feature_names_in_

        feature_base_map = {
            name: orig_feat
            for orig_feat in original_feature_names
            for name in encoded_feature_names
            if name.startswith(orig_feat + '_') or name == orig_feat
        }

        coefs = np.abs(model.coef_[0])
        feature_map_df = pd.DataFrame({
            'EncodedFeature': encoded_feature_names,
            'Importance': coefs
        })
        feature_map_df['OriginalFeature'] = feature_map_df['EncodedFeature'].map(feature_base_map)

        combined_importance = (
            feature_map_df.groupby("OriginalFeature")["Importance"]
            .sum()
            .reindex(features, fill_value=0)
        )

        plt.figure(figsize=(10, 6))
        bars = plt.barh(combined_importance.index, combined_importance.values)
        plt.xlabel("Total Coefficient Magnitude")
        plt.title(f"Feature Importance for Target: {target} (One per Original Feature)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va='center',
                ha='left',
                fontsize=9
            )

        if importance_filename:
            plt.savefig(importance_filename, format='jpg')
        plt.show()
    time.sleep(.1)

    return results

__docstrings__['run_logistic_model'] = run_logistic_model.__doc__


#### Helper function to check an existing saved file.
# Compares saved data to data in memory to decided if the file needs to be resaved or not
def save_if_changed(data, filename, verbose=True):
    file_path = Path(filename)

    # Detect and clean
    if isinstance(data, pd.DataFrame):
        data_clean = data.applymap(
            lambda v: v.tolist() if isinstance(v, (pd.Series, np.ndarray)) else v
        )
    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
        data_clean = pd.DataFrame([clean_nested_objects(d) for d in data])
    else:
        raise TypeError("Input data must be a DataFrame or list of dictionaries")

    # Check and save
    if file_path.exists():
        if verbose:
            print(f"{filename} exists. Checking for differences...")
        try:
            existing_data = pd.read_parquet(file_path)
            if data_clean.equals(existing_data):
                if verbose:
                    print("No changes detected. Skipping save.")
                return
            else:
                if verbose:
                    print("Changes detected. Overwriting saved file.")
        except Exception as e:
            if verbose:
                print(f"Error loading existing file, saving anyway: {e}")
    else:
        if verbose:
            print(f"{filename} not found. Saving...")

    data_clean.to_parquet(
        file_path,
        engine="pyarrow",
        compression="BROTLI",
        compression_level=11,
        index=False
    )
    if verbose:
        print(f"Saved: {filename}")

#### Used in EDA, outputs basic information about each feature
def resp_tally(df, colnames="all"):
    if colnames == "all":
        colnames = df.columns
    for col in colnames:
        print(f"\nValue counts for {col}")
        ig_counts = df[col].value_counts(dropna=False)
        print(ig_counts)




from IPython.display import display, Markdown
import pandas as pd

####Slightly different tally function
def resp_tally2(df, colnames="all", max_rows=10):
    if colnames == "all":
        colnames = df.columns

    tables = []
    for col in colnames:
        vc = df[col].value_counts(dropna=False)
        vc.index = vc.index.astype(str).fillna("NaN")
        vc = vc.head(max_rows)
        sub_df = pd.DataFrame({col: vc.index, f"{col}_count": vc.values})
        tables.append(sub_df.reset_index(drop=True))

    max_len = max(len(t) for t in tables)
    for i in range(len(tables)):
        tables[i] = tables[i].reindex(range(max_len)).reset_index(drop=True)

    final = pd.concat(tables, axis=1)
    display(final)
from pathlib import Path
import pandas as pd
import numpy as np

####Used to convert nested lists for saving as DataFrame
def clean_nested_objects(entry):
    def fix_value(v):
        if isinstance(v, pd.Series) or isinstance(v, np.ndarray):
            return v.tolist()
        return v
    return {k: fix_value(v) for k, v in entry.items()}


####Checks to see if the file exists, and if not saves it
def save_if_missing(data, file_path, name=""):
    """
    Save data to Parquet using p_save if the file doesn't exist.

    Args:
        data (list of dicts or DataFrame): Data to save.
        file_path (str or Path): Output file path.
        name (str): Optional label for print messages.
    """
    file_path = Path(file_path)

    label = f"[{name}] " if name else ""

    if file_path.exists():
        print(f"{label}File already exists: {file_path}")
        return

    print(f"{label}Save not found, saving to {file_path}...")

    # Clean and convert if not already a DataFrame
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        data_clean = [clean_nested_objects(item) for item in data]
        df = pd.DataFrame(data_clean)

    # Use the provided p_save function to write the file
    p_save(df)  # uses default params, but you can override if needed

    print(f"{label}Saved successfully.")
####Used in above function, or can be used on its own.
####Mainly created to save typing
def p_save(df, file_path=None, engine="pyarrow", compression="BROTLI", compression_level=11, index=False):
    if file_path is None:
        raise ValueError("file_path must be provided to p_save")
    df.to_parquet(
        file_path,
        engine=engine,
        compression=compression,
        compression_level=compression_level,
        index=index
    )
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

####Trains and runs tensorflow model
def run_tf_model(
    df_train,
    df_val,
    features,
    target,
    hidden_layers=[64, 32],
    dropout_rate=0.2,
    epochs=30,
    batch_size=32,
    encoder_drop=None,
    encoder_dtype=None,
    encoder_min_frequency=None,
    class_weight_mode='balanced'
):
    # One-hot encode features
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        drop=encoder_drop,
        dtype=encoder_dtype,
        min_frequency=encoder_min_frequency
    )
    X_train = encoder.fit_transform(df_train[features].astype(str))
    X_val = encoder.transform(df_val[features].astype(str))

    # Convert target to 0/1 if needed
    y_train = df_train[target]
    y_val = df_val[target]

    if y_train.dtype == 'object' or isinstance(y_train.iloc[0], str):
        unique_vals = sorted(y_train.dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Target must be binary. Found: {unique_vals}")
        target_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y_train = y_train.map(target_map)
        y_val = y_val.map(target_map)
        print(f"Target mapped as: {target_map}")
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')

    # Compute class weights
    if class_weight_mode == 'balanced':
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = {i: w for i, w in enumerate(weights)}
    else:
        class_weight = None

    # Build TensorFlow model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))

    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=class_weight
    )

    y_pred_probs = model.predict(X_val).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, zero_division=0)
    }

    print(f"TF Model Scores -> Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1_score']:.4f}")
    return results

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


### Runs and Trains random forest model
def run_rf_model(
    df_train,
    df_val,
    features,
    target,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    encoder_drop=None,
    encoder_dtype=None,
    encoder_min_frequency=None
):
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        drop=encoder_drop,
        dtype=encoder_dtype,
        min_frequency=encoder_min_frequency
    )

    X_train = encoder.fit_transform(df_train[features].astype(str))
    X_val = encoder.transform(df_val[features].astype(str))

    y_train = df_train[target]
    y_val = df_val[target]

    if y_train.dtype == 'object' or isinstance(y_train.iloc[0], str):
        unique_vals = sorted(y_train.dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Target must be binary. Found: {unique_vals}")
        target_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y_train = y_train.map(target_map)
        y_val = y_val.map(target_map)
        print(f"Target mapped as: {target_map}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',  # handle imbalance
        n_jobs=4
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, zero_division=0)
    }

    print(f"RF Validation Scores -> Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1_score']:.4f}")
    return results
