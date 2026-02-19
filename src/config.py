
DEBATE_TOPIC = "Should nations prioritize sustainable use over economic exploitation of natural resources?"
NUMBER_OF_REBUTTAL_ROUNDS = 2  

#Model configuration
DEFAULT_MODEL = "dolphin-phi:latest"
SUMMARY_MODEL = DEFAULT_MODEL

#RAG Configuration
KB_DIRECTORY = "./knowledge"          
VECTOR_STORE_PATH = "./chroma_db"     
EMBEDDING_MODEL = "nomic-embed-text" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENABLE_RAG = True
RETRIEVER_K = 3

# Agent Prompts (System Instructions) 
AGENT_SYSTEM_PROMPTS = {
    "DebateOrchestrator": (
        "You are a neutral debate moderator. Introduce the topic, call on speakers, maintain order, and conclude. "
        "Do not offer opinions."
    ),
    "Summarizer": (
        "You are a neutral summarization assistant. Summarize the debate history impartially, capturing key points from both sides."
    ),
    "DebateAgent": (
        "You are an AI debater in a structured debate. Present clear arguments for your stance and rebut respectfully. "
        "Use the provided summary and any relevant knowledge base context when available. Keep answers concise and structured."
    ),
    "AffirmativeAgent": (
        "You are an AI debater on the AFFIRMATIVE team arguing FOR the motion: '{topic}'."
    ),
    "NegativeAgent": (
        "You are an AI debater on the NEGATIVE team arguing AGAINST the motion: '{topic}'."
    ),
    "JudgeAgent": (
        "You are an AI judge observing the debate on: '{topic}'. "
        "Provide a brief, impartial summary of arguments from each side based only on the provided summary. Do not declare a winner."
    ),
}

# Stage Prompts 
STAGE_PROMPTS = {
    "opening": (
        "{retrieved_context}"
        "Give your opening statement for the debate topic:\n"
        "'{topic}'.\n\n"
        "Respond with 3–4 numbered, concise, evidence-based points."
    ),

    "rebuttal": (
        "{retrieved_context}"
        "Debate summary so far:\n{summary}\n\n"
        "Provide a rebuttal addressing the strongest opposing arguments.\n"
        "Respond with 2–3 numbered points."
    ),

    "closing": (
        "{retrieved_context}"
        "Debate summary so far:\n{summary}\n\n"
        "Provide a closing statement reinforcing your key arguments.\n"
        "Respond with 2–3 numbered points."
    ),

    "judge": (
        "Debate summary:\n{summary}\n\n"
        "Provide an impartial summary with the following format:\n\n"
        "Affirmative Key Points:\n- ...\n\n"
        "Negative Key Points:\n- ..."
    ),
}


# Summarization Prompt
SUMMARY_PROMPT_TEMPLATE = (
    "Please provide a concise, neutral summary of the following debate history. "
    "Include key arguments and rebuttals from both teams:\n\n{debate_history}\n\n"
    "Keep it short."
)

#Token Limits
MAX_TOKENS_PER_STAGE = {
    "opening": 280,
    "rebuttal": 240,
    "closing": 220,
    "judge": 200,
}
MAX_SUMMARY_TOKENS = 120


PROMPT_EXAMPLES = {
    "opening": [
        {
            "role": "user",
            "content": "Deliver an opening statement for: 'Should pineapple belong on pizza?' Provide 3-4 numbered points.",
        },
        {
            "role": "assistant",
            "content": (
                "Here is my opening statement:\n"
                "1. Pineapple adds sweet contrast to savory toppings.\n"
                "2. It balances saltiness and richness.\n"
                "3. Many cuisines pair fruit with savory successfully.\n"
                "4. Popular demand suggests broad appeal."
            ),
        },
    ]
}
