# src/agents.py

import ollama
from typing import Optional, List

from config import (
    AGENT_SYSTEM_PROMPTS, STAGE_PROMPTS,
    DEFAULT_MODEL, SUMMARY_MODEL,
    MAX_TOKENS_PER_STAGE, MAX_SUMMARY_TOKENS,
    ENABLE_RAG, RETRIEVER_K,
    SUMMARY_PROMPT_TEMPLATE
)
from debate_state import DebateState


def _ollama_chat(model: str, messages: List[dict], max_tokens: int) -> str:
    opts = {}
    if max_tokens and max_tokens > 0:
        opts["num_predict"] = max_tokens
    res = ollama.chat(model=model, messages=messages, options=opts)
    return res["message"]["content"].strip()


class BaseAgent:
    def __init__(self, name: str, role: str, model: str = DEFAULT_MODEL, retriever=None):
        self.name = name
        self.role = role
        self.model = model
        self.retriever = retriever
        self.system = AGENT_SYSTEM_PROMPTS.get(role, "")

    def _retrieve(self, query: str) -> str:
        if not (ENABLE_RAG and self.retriever):
            return "[No RAG enabled]\n\n"
        try:
            
            docs = self.retriever.invoke(query)
            if not docs:
                return "[No relevant information found]\n\n"
            formatted = []
            for d in docs[:RETRIEVER_K]:
                src = d.metadata.get("source", "N/A")
                formatted.append(f"Source: {src}\nContent: {d.page_content}")
            return "Relevant information from knowledge base:\n\n" + "\n---\n".join(formatted) + "\n\n"
        except Exception as e:
            return f"[RAG error: {e}]\n\n"

    def generate(self, user_prompt: str, stage: str, retrieved_context: str = "") -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        full_prompt = retrieved_context + user_prompt
        messages.append({"role": "user", "content": full_prompt})
        return _ollama_chat(self.model, messages, MAX_TOKENS_PER_STAGE.get(stage, 200))


class Debater(BaseAgent):
    def act(self, state: DebateState, stage: str, summary: Optional[str] = None) -> str:
        prompt_tpl = STAGE_PROMPTS[stage]
        query = f"Evidence relevant to: {state.topic}. Role={self.role}. Stage={stage}."
        if summary:
            query += f" Debate summary (excerpt): {summary[:250]}"
        retrieved = self._retrieve(query) if stage in ("opening", "rebuttal", "closing") else ""
        prompt = prompt_tpl.format(
            topic=state.topic,
            summary=summary or "",
            retrieved_context=retrieved
        )
        return self.generate(prompt, stage=stage, retrieved_context="")


class Judge(BaseAgent):
    
    def __init__(self, name: str = "Judge", model: str = DEFAULT_MODEL):
        super().__init__(name=name, role="JudgeAgent", model=model, retriever=None)

    def act(self, state: DebateState, summary: str) -> str:
        prompt = STAGE_PROMPTS["judge"].format(summary=summary)
        return self.generate(prompt, stage="judge")


class Orchestrator(BaseAgent):
    def __init__(self, state: DebateState, proponent: Debater, opponent: Debater, judge: Judge):
        super().__init__(name="Moderator", role="Moderator", model=DEFAULT_MODEL, retriever=None)
        self.state = state
        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge

    def summarize(self) -> str:
        history = self.state.as_text()
        prompt = SUMMARY_PROMPT_TEMPLATE.format(debate_history=history)
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPTS["Summarizer"]},
                    {"role": "user", "content": prompt}]
        return _ollama_chat(SUMMARY_MODEL, messages, MAX_SUMMARY_TOKENS)

    def run(self, rebuttal_rounds: int):
        yield {"type": "stage", "name": "Opening"}

        a1 = self.proponent.act(self.state, "opening")
        self.state.add(self.proponent.name, self.proponent.role, a1)
        yield {"type": "msg", "agent": self.proponent.name, "role": self.proponent.role, "text": a1}

        b1 = self.opponent.act(self.state, "opening")
        self.state.add(self.opponent.name, self.opponent.role, b1)
        yield {"type": "msg", "agent": self.opponent.name, "role": self.opponent.role, "text": b1}

        for i in range(rebuttal_rounds):
            yield {"type": "stage", "name": f"Rebuttal Round {i+1}"}
            s = self.summarize()
            yield {"type": "status", "text": "Summary generated."}

            ar = self.proponent.act(self.state, "rebuttal", summary=s)
            self.state.add(self.proponent.name, self.proponent.role, ar)
            yield {"type": "msg", "agent": self.proponent.name, "role": self.proponent.role, "text": ar}

            br = self.opponent.act(self.state, "rebuttal", summary=s)
            self.state.add(self.opponent.name, self.opponent.role, br)
            yield {"type": "msg", "agent": self.opponent.name, "role": self.opponent.role, "text": br}

        yield {"type": "stage", "name": "Closing"}
        s2 = self.summarize()

        ac = self.proponent.act(self.state, "closing", summary=s2)
        self.state.add(self.proponent.name, self.proponent.role, ac)
        yield {"type": "msg", "agent": self.proponent.name, "role": self.proponent.role, "text": ac}

        bc = self.opponent.act(self.state, "closing", summary=s2)
        self.state.add(self.opponent.name, self.opponent.role, bc)
        yield {"type": "msg", "agent": self.opponent.name, "role": self.opponent.role, "text": bc}

        yield {"type": "stage", "name": "Judge Summary"}
        j = self.judge.act(self.state, summary=s2)
        yield {"type": "msg", "agent": self.judge.name, "role": self.judge.role, "text": j}

        yield {"type": "done"}
