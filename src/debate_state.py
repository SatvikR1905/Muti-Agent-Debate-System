# src/debate_state.py

class DebateState:
    def __init__(self, topic: str):
        self.topic = topic
        self.history = []  # [{agent, role, text}]

    def add(self, agent: str, role: str, text: str):
        self.history.append({"agent": agent, "role": role, "text": text})

    def as_text(self) -> str:
        out = [f"Debate Topic: {self.topic}", "", "-- Debate History --"]
        if not self.history:
            out.append("No arguments yet.")
        else:
            for item in self.history:
                out.append(f"[{item['role']} - {item['agent']}]\n{item['text']}\n")
        out.append("-- End --")
        return "\n".join(out)
