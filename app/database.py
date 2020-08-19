import mysql.connector
import pandas as pd
import os
from ast import literal_eval
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(dotenv_path=Path("..") / ".env")
mydb = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    db=os.getenv("DB_NAME"),
    password=os.getenv("DB_PASS")
)
cursor = mydb.cursor()


def db_get_faq_feedback():
    """
    :return: user utterances resulting in positive and negative feedback
    """
    cursor.execute("SELECT F.id, F.utterance, F.correct, F.faq_id, M.market "
                   "FROM faq_feedback_multilg F, markets M "
                   "WHERE F.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    pos_feedback = result[result['correct'] == 1][['utterance', 'market', 'faq_id']]
    neg_feedback = result[result['correct'] == -1][['utterance', 'market', 'faq_id']]
    return pos_feedback, neg_feedback


def db_get_message_analytics(something_else):
    """
    :param something_else: return only something else triggers if true, otherwise return all text
    :return: user utterances from message_analytics
    """
    cursor.execute("SELECT ts_in_db, top_intent, user_event, market, conversation_id "
                   "FROM message_analytics MS "
                   "LEFT JOIN  markets M ON MS.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    if something_else:
        something_else = result[result['top_intent'] == 'navigational:something_else']
        idx_prev = something_else.index - 1
        something_else_triggers = result.iloc[idx_prev, :]
        text = something_else_triggers['user_event'].map(eval)
        result = something_else_triggers
    else:
        user_event_list = result['user_event'].to_list()
        user_event_list = str(user_event_list).replace("null", "123456")
        result['user_event'] = literal_eval(user_event_list)
        text = result['user_event'].map(eval)

    text = text.apply(pd.Series)
    text['market'] = result['market']
    text['ts_in_db'] = result['ts_in_db']
    text['conversation_id'] = result['conversation_id']
    return text


