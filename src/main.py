# src/main.py

from config import DEBATE_TOPIC, NUMBER_OF_REBUTTAL_ROUNDS, ENABLE_RAG
from debate_state import DebateState
from rag_pipeline import index_knowledge_base, get_retriever
from agents import Debater, Judge, Orchestrator


def main():
    retriever = None
    if ENABLE_RAG:
        vs = index_knowledge_base(
            kb_directory="./knowledge",
            vector_store_path="./chroma_db",
            embedding_model="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50
        )
        retriever = get_retriever(vs) if vs else None

    state = DebateState(DEBATE_TOPIC)

    pro = Debater(name="Proponent", role="Proponent", retriever=retriever)
    opp = Debater(name="Opponent", role="Opponent", retriever=retriever)
    judge = Judge(name="Judge")

    orch = Orchestrator(state, pro, opp, judge)

    for event in orch.run(NUMBER_OF_REBUTTAL_ROUNDS):
        if event["type"] == "stage":
            print(f"\n=== {event['name']} ===\n")
        elif event["type"] == "status":
            print(f"[{event['text']}]")
        elif event["type"] == "msg":
            print(f"{event['agent']} ({event['role']}):\n{event['text']}\n")
        elif event["type"] == "done":
            print("\n✅ Debate complete.")


if __name__ == "__main__":
    main()
