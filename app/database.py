import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
  
)
cursor = mydb.cursor()


def db_get_faq_feedback():
    cursor.execute("SELECT * FROM faq_feedback_multilg")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    pos_feedback = result[result['correct'] == 1][['utterance']]
    neg_feedback = result[result['correct'] == -1][['utterance']]
    return pos_feedback, neg_feedback


def db_get_something_else_triggers():
    cursor.execute("SELECT * FROM message_analytics")
    result = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])

    something_else = result[
        result['top_intent'] == 'navigational:something_else']
    idx_prev = something_else.index - 1
    something_else_triggers = result.iloc[idx_prev, :]

    text = something_else_triggers['user_event'].map(eval)
    text = text.apply(pd.Series)
    text = text[text['text'] != 'navigational:restart'][['text']]
    return text
