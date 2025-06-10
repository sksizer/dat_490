import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

__docstrings__ = {}

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
                       n_init=5, verbose=1, cluster_col_name=None):
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
        DataFrame with cluster column added.
    """
    print(f"Clustering on {len(feature_cols)} features: {feature_cols}")
    X_cluster = df[feature_cols].astype(str)

    km = KModes(n_clusters=n_clusters, init=init_method, n_init=n_init, verbose=verbose)
    clusters = km.fit_predict(X_cluster)

    df = df.copy()
    if cluster_col_name is None:
        init_code = 'h' if init_method.lower() == 'huang' else 'c'
        cluster_col_name = f'kmode_n{n_clusters}_i{n_init}_{init_code}'

    df[cluster_col_name] = clusters
    return df

__docstrings__['run_kmodes_cluster'] = run_kmodes_cluster.__doc__


# TF+KMeans Clustering
def run_tf_clustering(df, feature_cols, n_clusters=5, latent_dim=8, cluster_col_name=None,
                      epochs=20, batch_size=512, verbose=1):
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

    input_dim = X_encoded.shape[1]
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(64, activation='relu')(inputs)
    bottleneck = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = tf.keras.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_encoded, X_encoded, epochs=epochs, batch_size=batch_size, verbose=verbose)

    encoder_model = tf.keras.Model(inputs, bottleneck)
    latent_features = encoder_model.predict(X_encoded)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features)

    if cluster_col_name is None:
        cluster_col_name = f"tf_n{n_clusters}_d{latent_dim}_e{epochs}"

    df = df.copy()
    df[cluster_col_name] = cluster_labels
    return df,cluster_col_name

__docstrings__['run_tf_clustering'] = run_tf_clustering.__doc__


# Cluster Overlap Plot
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

__docstrings__['plot_cluster_overlap'] = plot_cluster_overlap.__doc__


# Feature Distribution Comparison
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
    encoder_min_frequency=None
):
    """
    Train and evaluate a logistic regression classifier using one-hot encoding.

    Args:
        df_train (DataFrame): Training set.
        df_val (DataFrame): Validation set.
        features (list): List of categorical feature column names.
        target (str): Target column name.
        C (float): Inverse of regularization strength. Smaller values specify stronger regularization.
        penalty (str): Norm used in the penalization ('l1', 'l2', 'elasticnet', or 'none').
        solver (str): Algorithm to use in the optimization problem ('liblinear', 'lbfgs', etc.).
        max_iter (int): Maximum number of iterations taken for the solvers to converge.
        encoder_drop (str or None): Strategy to drop one of the categories per feature ('first', 'if_binary', or None).
        encoder_dtype (np.dtype or None): Desired dtype of the encoded output.
        encoder_min_frequency (int, float, or None): Minimum frequency for a category to be included.

    Returns:
        float: Accuracy on the validation set.
    """
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

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val))
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy
__docstrings__['run_logistic_model'] = run_logistic_model.__doc__