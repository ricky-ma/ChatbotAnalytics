import mysql.connector
import pandas as pd
from ast import literal_eval

mydb = mysql.connector.connect(
    
)
cursor = mydb.cursor()


def db_get_faq_feedback():
    cursor.execute("SELECT F.id, F.utterance, F.correct, F.faq_id, M.market "
                   "FROM faq_feedback_multilg F, markets M "
                   "WHERE F.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    pos_feedback = result[result['correct'] == 1][['utterance', 'market', 'faq_id']]
    neg_feedback = result[result['correct'] == -1][['utterance', 'market', 'faq_id']]
    return pos_feedback, neg_feedback


def db_get_message_analytics(something_else):
    cursor.execute("SELECT ts_in_db, top_intent, user_event, market "
                   "FROM message_analytics MS "
                   "LEFT JOIN  markets M ON MS.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    if something_else:
        something_else = result[result['top_intent'] == 'navigational:something_else']
        idx_prev = something_else.index - 1
        something_else_triggers = result.iloc[idx_prev, :]

        something_else_txt = something_else_triggers['user_event'].map(eval)
        something_else_txt = something_else_txt.apply(pd.Series)
        something_else_txt['market'] = something_else_triggers['market']
        something_else_txt['ts_in_db'] = something_else_triggers['ts_in_db']
        return something_else_txt
    else:
        user_event_list = result['user_event'].to_list()
        user_event_list = str(user_event_list).replace("null", "123456")
        result['user_event'] = literal_eval(user_event_list)
        all_txt = result['user_event'].map(eval)
        all_txt = all_txt.apply(pd.Series)
        all_txt['market'] = result['market']
        all_txt['ts_in_db'] = result['ts_in_db']
        return all_txt


