# Chatbot Evaluation and Maintenance
Exploratory data analysis and interactive model-understanding and 
evaluation tool for chatbot training data and feedback.

The tool is built to answer questions such as:
- What examples does my model perform poorly on?
    - In terms of classification?
    - In terms of user feedback?
- Can user feedback be attributed to adversarial behaviour?
- Is there mislabeled text in the training set?

## Features
The tool runs through a browser-based dashboard. Standard features 
include:
- A visualization of the low-dimensional representation of the 
embeddings for both user queries as well as any stored data.
- Aggregate analysis on chatbot metrics:
    - language
    - feedback type (upvote, downvote, something else, none)
    - chatbot FAQ ID
    - confidence of top intent
    - outlier scores for training data
    - novelty scores for feedback data
- Future metrics to be added include:
    - ranking of delivered content from bot API
    - visible ranking when presented to users
    - timestamp
    - website
    - IP address
    - session ID
    - attach policies (set of rules governing a chatbot)
    - user annotated FAQ ID
    - distance of query from FAQ ID

## Feedback Workflows
Workflow templates provide standard solutions for chatbot performance 
evaluation and maintenance. 
- Confirmation that upvote user feedback agrees with chatbot 
predictions.
    - If user labels agree with chatbot labels, then feedback is 
    likely genuine and can be quickly added to training data.
    - If user labels are in many different categories, then the 
    classifier performed poorly, but user says the result is correct. 
    The examples should be checked before being added to training data.
    - If user labels are in another category, then feedback can be
    attributed to adversarial behaviour.
- Confirmation that downvote user feedback agrees with chatbot 
predictions.
    - If user labels agree with chatbot labels, then feedback can be 
    attributed to adversarial behaviour.
    - If user labels are in many different categories, then the
    classifier performed poorly and the user confirms this. The 
    examples should be checked and reviewed.
    - If user labels are in another category, then feedback is likely
    genuine, but needs confirmation. The examples should be checked 
    before being added to training data.
- Confirmation that "something else" button presses agree with chatbot
predictions.
    - When a user chooses something else, confidences of previously
    returned chatbot intents should be low.
    - Examples should be checked before deciding whether to add data
    to an existing category or a new category.
    

