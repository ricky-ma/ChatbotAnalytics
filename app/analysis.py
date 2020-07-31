import pandas as pd
import numpy as np
import umap
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder


def load_data(df_vec, df_meta):
    reducer = umap.UMAP(n_components=2)
    # reducer = umap.UMAP(n_components=3)

    data = df_vec[df_vec.columns].values
    print("Scaling data")
    scaled_data = StandardScaler().fit_transform(data)
    print("Fitting to UMAP")
    embedding = reducer.fit_transform(scaled_data, y=df_meta['FAQ_id'])
    print("Embedding shape: " + str(embedding.shape))
    embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
    # embedding_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
    final_df = embedding_df.join(df_meta)
    return scaled_data, final_df


def get_outliers(raw_data, embedded_data):
    lof = LocalOutlierFactor(n_neighbors=10, contamination='auto')
    # Fit LOF on raw data
    outlier_scores = lof.fit_predict(raw_data)

    joined = embedded_data.join(pd.DataFrame(outlier_scores, columns=['outlier_score']))
    joined['negative_outlier_factor'] = lof.negative_outlier_factor_

    outliers = joined.loc[joined['outlier_score'] == -1]
    outlier_cats = outliers.FAQ_id.unique()

    final_df = joined[joined['FAQ_id'].isin(list(outlier_cats))]
    final_df.loc[final_df['outlier_score'] == -1, 'outlier_score'] = 4
    final_df['FAQ_id'] = pd.to_numeric(final_df['FAQ_id'])
    return final_df


def reduce(dataframe, reducer='umap', n_comp=80):
    if reducer == 'pca':
        reducer = PCA(n_components=n_comp)
    else:
        reducer = umap.UMAP(n_components=n_comp)
    # labels = pd.DataFrame(dataframe).join(metadata)['FAQ_id']
    # scaled_data = StandardScaler().fit_transform(dataframe)
    # embedding = reducer.fit_transform(scaled_data)
    # labels = dataframe['category']
    scaled_data = StandardScaler().fit_transform(dataframe.loc[:, '1'::])
    embedding = reducer.fit_transform(scaled_data)
    embedding = pd.DataFrame(embedding)
    return embedding


def autoencode(train_data, test_data):
    X_train = reduce(train_data)
    X_test = reduce(test_data)
    clf1 = AutoEncoder(hidden_neurons=[25, 15, 10, 2, 10, 15, 25])
    clf1.fit(X_train)

    # Get the outlier scores for the train data
    y_train_scores = clf1.decision_scores_

    # Predict the anomaly scores
    y_test_scores = clf1.decision_function(X_test)  # outlier scores
    y_test_scores = pd.Series(y_test_scores)
    return X_test, y_test_scores


def get_novel(X_test, y_test_scores, decision):
    df_test = X_test.copy()
    df_test['score'] = y_test_scores

    # Lower score is non-novel, higher score is novel
    # --> 0: non-novel, 1: novel
    df_test['cluster'] = np.where(df_test['score'] < decision, 0, 1)
    df_test['cluster'].value_counts()
    # df_test.groupby('cluster').mean()

    novel_df = df_test[df_test['cluster'] == 1]
    return novel_df


def cluster_novel(novel_df):
    novel_df = novel_df.drop(columns=['cluster', 'score'])
    # TODO: implement automated clustering algorithm to find clusters
