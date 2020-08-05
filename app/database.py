import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
    
)
cursor = mydb.cursor()


def db_get_faq_feedback():
    cursor.execute("SELECT F.utterance, F.correct, M.market "
                   "FROM faq_feedback_multilg F, markets M "
                   "WHERE F.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    pos_feedback = result[result['correct'] == 1][['utterance', 'market']]
    neg_feedback = result[result['correct'] == -1][['utterance', 'market']]
    return pos_feedback, neg_feedback


def db_get_message_analytics():
    cursor.execute("SELECT top_intent, user_event, market "
                   "FROM message_analytics MS "
                   "LEFT JOIN  markets M ON MS.market_id = M.market_id")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    print(result.head)
    something_else = result[result['top_intent'] == 'navigational:something_else']
    idx_prev = something_else.index - 1
    something_else_triggers = result.iloc[idx_prev, :]

    something_else_txt = something_else_triggers['user_event'].map(eval)
    something_else_txt = something_else_txt.apply(pd.Series)
    something_else_txt['market'] = something_else_triggers['market']
    return something_else_txt
