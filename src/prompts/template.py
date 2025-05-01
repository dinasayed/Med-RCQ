import random


#---------------------------------Inference Prompts-----------------------------

# Invoke MedConclusion to generate prompt:
PROMPT='''You are a helpful medical assistant. Write a conclusion for the following article:\n
Title: _TITLE_
_CONTEXT_
Conclusion:'''

# Invoke MedQA to make decision according to PubMedQA annotation criteria:
PROMPT='''You are a helpful medical assistant. I will give you a context and a question of a study, based on the context you will answer the question by Yes or No or Maybe. In order to answer correctly you will analyze the study findings and results carefully.
If the outcomes are significant or evidences are strong and results are mostly leaning towards agreeing with the question then answer with Yes.
If the outcomes are insignificant or evidences are weak and results are mostly against the question by refuting it then answer with No.
If the outcomes are tie between Yes and No, then answer with Maybe
Think carefully.
Context: _CONTEXT_
Question: _QUESTION_
Final Answer:'''

# Using GPT4.1 Nano to choose best answer from Multi-choice question:
PROMPT = '''You are a medical assistant answering multiple-choice questions. Read the context and question carefully, and answer by selecting the **letter** of the best option (A, B, C, or D). Just return the letter, nothing else.

Context:_CONTEXT_

Question:_QUESTION_

Options:_OPTIONS_

Answer:'''

# Using GPT4.1 Nano generate short title:
PROMPT
'''You are given a question and an explanation from a medical exam.
Your task is to generate a short, clear, and descriptive title based on the content.
Be concise (5-10 words max).

Question:_QUESTION_
Explanation:_CONTEXT_

Title:'''

#---------------------------------Training Prompts------------------------------

# For MedConclusion return a random instruction for generating conclusion
def get_random_systeminst():

    sysinst_sentences=[
        "You are a helpful medical assistant. Write a conclusion for the following article:\n",
        "You are a helpful medical assistant. Write a conclusion for the following study:\n",
        "You are a helpful medical assistant. Generate a conclusion for the following article:\n",
        "You are a helpful medical assistant. Generate a conclusion for the following study:\n",
        "You are a helpful medical assistant. Write a conclusion for the following research study:\n",
        "You are a helpful medical assistant. Generate a conclusion for the following research study:\n",
        "You are a helpful medical assistant. Write a conclusion for the following medical study:\n",
        "You are a helpful medical assistant. Write a conclusion for the following medical article:\n",
        "You are a helpful medical assistant. Conclude the following study:\n",
        "You are a helpful medical assistant. Conclude the following article:\n",
        "You are a helpful medical assistant. Write down a conclusion for the following article:\n"
    ]

    return random.choice(sysinst_sentences)