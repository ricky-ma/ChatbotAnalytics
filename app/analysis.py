import pandas as pd
import umap
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow_hub as hub
from app.database import db_get_faq_feedback, db_get_message_analytics

pos_feedback, neg_feedback = db_get_faq_feedback()
something_else_triggers = db_get_message_analytics(something_else=True)
all_messages = db_get_message_analytics(something_else=False)


def load_data(df_vec, df_meta):
    """
    :param df_vec: dataframe of vector embeddings of text
    :param df_meta: dataframe containing corresponding metadata
    :return: scaled vector embeddings, 2D UMAP embedding w/ metadata
    """
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
    """
    :param text: raw text data
    :return: dataframe of vector embeddings
    """
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)
    vecs = embed(text)
    return pd.DataFrame(vecs)


def reduce(dataframe, n_comp=200):
    """
    :param dataframe: dataframe of vector embeddings of text
    :param n_comp: number of components to reduce down to
    :return: reduced vector embeddings, reducer
    """
    reducer = PCA(n_components=n_comp)
    scaled_data = StandardScaler().fit_transform(dataframe)
    embedding = reducer.fit_transform(scaled_data)
    embedding = pd.DataFrame(embedding)
    return embedding, reducer


def get_outliers(embedded_df, reduced_df):
    """
    :param embedded_df: full vector embeddings of text
    :param reduced_df: reduced vector embeddings of text
    :return: dataframe containing x, y-coords, outlier score, and corresponding metadata
    """
    lof = LocalOutlierFactor(n_neighbors=10, contamination='auto')
    # Fit LOF on raw data
    outlier_scores = lof.fit_predict(embedded_df)

    joined = reduced_df.join(pd.DataFrame(outlier_scores, columns=['outlier_score']))
    joined['negative_outlier_factor'] = lof.negative_outlier_factor_

    outliers = joined.loc[joined['outlier_score'] == -1]
    outlier_cats = outliers.FAQ_id.unique()

    final_df = joined[joined['FAQ_id'].isin(list(outlier_cats))]
    final_df.loc[final_df['outlier_score'] == -1, 'outlier_score'] = 4
    final_df['FAQ_id'] = pd.to_numeric(final_df['FAQ_id'])
    return final_df


def get_novelties(train_data, something_else, pos, neg):
    """
    :param train_data: vector embeddings of text train data
    :param something_else: vector embeddings of "something else" user utterances
    :param pos: vector embeddings of user utterances resulting in positive feedback
    :param neg: vector embeddings of user utterances resulting in negative feedback
    :return: novelty scores of all data passed in
    """
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
    clf.fit(train_data)
    y_train_scores = clf.negative_outlier_factor_
    y_train_scores = pd.DataFrame(y_train_scores, columns=['score'])
    y_train_scores['dataset'] = 'train'

    something_else_scores = clf.score_samples(something_else)  # outlier scores
    something_else_scores = pd.Series(something_else_scores, name='score')
    something_else_scores = something_else_scores.to_frame()
    something_else_scores['dataset'] = 'something else'

    pos_scores = clf.score_samples(pos)  # outlier scores
    pos_scores = pd.Series(pos_scores, name='score')
    pos_scores = pos_scores.to_frame()
    pos_scores['dataset'] = 'positive feedback'

    neg_scores = clf.score_samples(neg)  # outlier scores
    neg_scores = pd.Series(neg_scores, name='score')
    neg_scores = neg_scores.to_frame()
    neg_scores['dataset'] = 'negative feedback'

    scores = pd.concat([y_train_scores, something_else_scores, pos_scores, neg_scores]).reset_index(drop=True)
    return scores


def get_novel_scores(something_else, pos, neg):
    """
    :param something_else: "something else" user utterances
    :param pos: user utterances resulting in positive feedback
    :param neg: user utterances resulting in negative feedback
    :return: novelty scores of all data passed in
    """
    print("Embedding text...")
    something_else_vecs = embed_text(something_else)
    pos_vecs = embed_text(pos)
    neg_vecs = embed_text(neg)

    # replace with actual trained dataset
    # train_vecs = pd.read_csv("./data/extracted_n26_tsv_vecs.tsv", delimiter='\t|,', header=None, engine='python')
    train_vecs = pd.read_csv("C:/Users/mrric/!Projects/Algomo/ChatbotAnalytics/data/extracted_n26_tsv_vecs.tsv",
                             delimiter='\t|,', header=None, engine='python')
    train_vecs = train_vecs.drop(train_vecs.columns[0], axis=1)

    print("Fitting LOF...")
    scores = get_novelties(train_vecs, something_else_vecs, pos_vecs, neg_vecs)
    return scores


def novel_df():
    """
    :return: dataframe containing novelty scores of user feedback w/ metadata
    """
    novelty_scores = get_novel_scores(something_else_triggers['text'], pos_feedback['utterance'],
                                      neg_feedback['utterance'])
    txt_frames = [something_else_triggers['text'], pos_feedback['utterance'], neg_feedback['utterance']]
    mkt_frames = [something_else_triggers['market'], pos_feedback['market'], neg_feedback['market']]

    novel = pd.DataFrame()
    novel['score'] = novelty_scores[novelty_scores['dataset'] != 'train']['score'].reset_index(drop=True)
    novel['dataset'] = novelty_scores[novelty_scores['dataset'] != 'train']['dataset'].reset_index(drop=True)
    novel['market'] = pd.concat(mkt_frames).reset_index(drop=True)
    novel['text'] = pd.concat(txt_frames).reset_index(drop=True)

    novel['top intent'] = None
    novel['confidence'] = None
    novel['top intent'].loc[novel['dataset'] == 'positive feedback'] = pos_feedback['faq_id']
    novel['top intent'].loc[novel['dataset'] == 'negative feedback'] = neg_feedback['faq_id']
    top_intents = [sub['intent'] for sub in something_else_triggers['top_intent']]
    confidences = [sub['confidence'] for sub in something_else_triggers['top_intent']]
    novel['top intent'].loc[novel['dataset'] == 'something else'] = top_intents
    novel['confidence'].loc[novel['dataset'] == 'something else'] = confidences
    return novel


def analyze_mkts(novel):
    """
    :param novel: dataframe containing novelty scores of user feedback w/ metadata
    :return: dataframe containing various analytics for each market
    """
    analysis = []
    for market in novel['market'].unique():
        mkt_cond = novel['market'] == market
        num_pos = len(novel[(novel['dataset'] == 'positive feedback') & mkt_cond])
        num_neg = len(novel[(novel['dataset'] == 'negative feedback') & mkt_cond])
        num_something_else = len(novel[(novel['dataset'] == 'something else') & mkt_cond])
        avg_confidence = novel[mkt_cond]['confidence'].mean()
        avg_novelty = novel[mkt_cond]['score'].mean()
        analysis.append((market, num_pos, num_neg, num_something_else, avg_confidence, avg_novelty))
    analysis_df = pd.DataFrame(analysis, columns=['market', 'positive feedbacks', 'negative feedbacks',
                                                  'something else', 'avg top intent confidence', 'avg novelty score'])
    return analysis_df


def confidence_over_time():
    """
    :return: dataframe containing average weekly confidence of the top intents
    """
    confidence_map = []
    for intent_dict in all_messages['top_intent']:
        if 'confidence' in intent_dict.keys():
            confidence_map.append(True)
        else:
            confidence_map.append(False)
    x = all_messages[confidence_map]
    df = pd.DataFrame()
    top_intents = [sub['intent'] for sub in x['top_intent']]
    confidences = [sub['confidence'] for sub in x['top_intent']]
    df['top intent'] = top_intents
    df['confidence'] = pd.to_numeric(confidences)
    # df['market'] = something_else_triggers['market']

    df['timestamp'] = x['ts_in_db'].reset_index(drop=True)
    df = df.groupby(pd.Grouper(key="timestamp", freq="1W")).mean()
    df['timestamp'] = df.index
    return df

