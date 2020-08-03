import pandas as pd
import numpy as np
import umap
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow_hub as hub


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


def embed_text(text):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)
    vecs = embed(text)
    return pd.DataFrame(vecs)


def reduce(dataframe, n_comp=200):
    reducer = PCA(n_components=n_comp)
    scaled_data = StandardScaler().fit_transform(dataframe)
    embedding = reducer.fit_transform(scaled_data)
    embedding = pd.DataFrame(embedding)
    return embedding, reducer


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


def get_novelties(train_data, test_data):
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
    clf.fit(train_data)
    y_train_scores = clf.negative_outlier_factor_
    y_train_scores = pd.DataFrame(y_train_scores, columns=['score'])
    y_train_scores['dataset'] = 'train'

    #     y_pred_test = clf.predict(test_data)
    y_test_scores = clf.score_samples(test_data)  # outlier scores
    y_test_scores = pd.Series(y_test_scores, name='score')
    y_test_scores = y_test_scores.to_frame()
    y_test_scores['dataset'] = 'test'
    return test_data, pd.concat([y_train_scores, y_test_scores]).reset_index(drop=True)


def get_novel_scores(novel):
    print("Embedding text...")
    novel_vecs = embed_text(novel)
    print("Reducing components...")
    # novel_vecs = reduce(novel_vecs)

    # replace with actual trained dataset
    train_vecs = pd.read_csv("./data/extracted_n26_tsv_vecs.tsv", delimiter='\t|,', header=None, engine='python')
    train_vecs = train_vecs.drop(train_vecs.columns[0], axis=1)
    # train_vecs = reduce(train_vecs)

    print("Fitting LOF...")
    test_data, scores = get_novelties(train_vecs, novel_vecs)
    return test_data, scores


def cluster_novel(novel_df):
    novel_df = novel_df.drop(columns=['cluster', 'score'])
    # TODO: implement automated clustering algorithm to find clusters
