PROMPT_EVAL = """
Below is some task instruction and a corresponding response to evaluate.
Your job is to answer whether the specified constraint is satisfied or not.

###Task instruction:
{instruction}

###Response:
{response}

###Question:
Is the following constraint satisfied: "{constraint}"?

###Answer:
""".strip()