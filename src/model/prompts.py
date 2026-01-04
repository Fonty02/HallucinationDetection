


SYSTEM_PROMPT_BB="""
    You are an expert in several domains. You'll be provided with a fact, and your task is to confirm or deny its truthfulness.
    Do not provide any additional information or explanations.
    Just answer only with ''yes'' or ''no''.
"""

USER_PROMPT_BB="""Is the fact true? Fact: {question}
    Answer: """

SYSTEM_PROMPT_HE="""
    You are an expert in several domains. You'll be provided with a question, and your task is to provide the correct answer.
    Do not provide any additional information or explanations.
    Just answer the question directly.
"""

USER_PROMPT_HE=""" Question: {question}
    Answer: 
    """