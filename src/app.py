# src/app.py
import time
import streamlit as st
from collections import deque

from config import (
    DEBATE_TOPIC,
    NUMBER_OF_REBUTTAL_ROUNDS,
    ENABLE_RAG,
    KB_DIRECTORY,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
)

from rag_pipeline import index_knowledge_base, get_retriever
from agents import Debater, Judge, Orchestrator
from debate_state import DebateState


st.set_page_config(page_title="Multi-Agent Debate System", layout="wide")


def ss_init():
    if "status" not in st.session_state:
        st.session_state.status = "Configure and start the debate."
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=500)
    if "timeline" not in st.session_state:
        st.session_state.timeline = deque(maxlen=60)
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {"Affirmative": "Idle", "Negative": "Idle", "Judge": "Idle"}
    if "running" not in st.session_state:
        st.session_state.running = False

    if "topic" not in st.session_state:
        st.session_state.topic = DEBATE_TOPIC
    if "rounds" not in st.session_state:
        st.session_state.rounds = NUMBER_OF_REBUTTAL_ROUNDS
    if "enable_rag" not in st.session_state:
        st.session_state.enable_rag = ENABLE_RAG


ss_init()



st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem !important; }

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 700px at 30% 10%, #0b1733 0%, #050812 55%, #040510 100%) !important;
}

.card {
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(2,6,23,0.55);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.arena-title { font-size: 30px; font-weight: 900; margin: 6px 0 6px 0; }
.subtle { color: rgba(148,163,184,0.95); }

.bubble {
  border-radius: 14px;
  padding: 12px 12px;
  margin: 10px 0;
  border: 1px solid rgba(148,163,184,0.18);
  line-height: 1.45;
}
.bubble h4 { margin: 0 0 8px 0; font-size: 15px; }
.bubble p { margin: 0; white-space: pre-wrap; }

.green { background: rgba(34,197,94,0.13); border-color: rgba(34,197,94,0.35); }
.red   { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.33); }
.purple{ background: rgba(168,85,247,0.12); border-color: rgba(168,85,247,0.35); }

.badge { display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 10px; vertical-align: middle; }
.badge-green { background: #22c55e; }
.badge-red { background: #ef4444; }
.badge-purple { background: #a855f7; }

.timeline li { margin: 6px 0; color: rgba(203,213,225,0.95); }
.timeline .muted { color: rgba(148,163,184,0.95); font-style: italic; }
</style>
""",
    unsafe_allow_html=True,
)



def push_status(msg: str):
    st.session_state.status = msg
    st.session_state.history.append({"type": "status", "text": msg})


def push_timeline(msg: str):
    st.session_state.timeline.appendleft(msg)


def push_message(name: str, role: str, text: str):
    st.session_state.history.append({"type": "msg", "name": name, "role": role, "text": text})


def build_orchestrator(topic: str, rounds: int, enable_rag: bool):
    retriever = None
    if enable_rag:
        vs = index_knowledge_base(
            kb_directory=KB_DIRECTORY,
            vector_store_path=VECTOR_STORE_PATH,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        retriever = get_retriever(vs, k=RETRIEVER_K) if vs else None

    state = DebateState(topic=topic)
    affirmative = Debater(name="Affirmative", role="AffirmativeAgent", retriever=retriever)
    negative = Debater(name="Negative", role="NegativeAgent", retriever=retriever)
    judge = Judge(name="Judge")

    orch = Orchestrator(state=state, proponent=affirmative, opponent=negative, judge=judge)
    return orch.run(rebuttal_rounds=rounds)


def render_agents_panel():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Agents")
    st.markdown(
        f"<div><span class='badge badge-green'></span><b>Affirmative</b><br/><span class='subtle'>Status: {st.session_state.agent_status['Affirmative']}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div><span class='badge badge-red'></span><b>Negative</b><br/><span class='subtle'>Status: {st.session_state.agent_status['Negative']}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div><span class='badge badge-purple'></span><b>Judge</b><br/><span class='subtle'>Status: {st.session_state.agent_status['Judge']}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"**System**<br/><span class='subtle'>{st.session_state.status}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline_and_judge():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Stage & Judge")
    st.markdown("<div class='subtle' style='margin-bottom:8px'>Timeline</div>", unsafe_allow_html=True)

    if not st.session_state.timeline:
        st.markdown("<div class='subtle'><i>No events yet.</i></div>", unsafe_allow_html=True)
    else:
        st.markdown("<ul class='timeline'>", unsafe_allow_html=True)
        for item in list(st.session_state.timeline)[:12]:
            cls = "muted" if ("summary" in item.lower() or "generated" in item.lower()) else ""
            st.markdown(f"<li class='{cls}'>{item}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    last_judge = None
    for it in reversed(st.session_state.history):
        if it.get("type") == "msg" and it.get("name") == "Judge":
            last_judge = it
            break

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Judge</div>", unsafe_allow_html=True)

    if last_judge:
        st.markdown(
            f"""
            <div class="bubble purple">
              <h4>⚖️ Judge ({last_judge['role']})</h4>
              <p>{last_judge['text']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='subtle'><i>No judge output yet.</i></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_side(column_name: str, color_class: str):
    msgs = [x for x in st.session_state.history if x.get("type") == "msg" and x.get("name") == column_name]
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    badge = "badge-green" if column_name == "Affirmative" else "badge-red"
    st.markdown(
        f"<span class='badge {badge}'></span><b>{column_name}</b>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not msgs:
        st.markdown("<div class='subtle'><i>Waiting for first turn…</i></div>", unsafe_allow_html=True)
    else:
        for m in msgs[-6:]:
            st.markdown(
                f"""
                <div class="bubble {color_class}">
                  <h4>{column_name} ({m['role']})</h4>
                  <p>{m['text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_arena():
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
          <div class="arena-title">Debate Arena</div>
          <div class="subtle">❝ {st.session_state.topic} ❞</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, mid, n = st.columns([1.25, 0.9, 1.25], gap="large")
    with a:
        render_side("Affirmative", "green")
    with mid:
        render_timeline_and_judge()
    with n:
        render_side("Negative", "red")



with st.sidebar:
    st.markdown("## Controls")
    topic = st.text_area("Debate Topic", st.session_state.topic, height=90)
    rounds = st.slider("Rebuttal rounds", 0, 5, int(st.session_state.rounds))
    enable_rag = st.checkbox("Enable RAG (Knowledge Base)", bool(st.session_state.enable_rag))

    st.session_state.topic = topic
    st.session_state.rounds = rounds
    st.session_state.enable_rag = enable_rag

    c1, c2 = st.columns(2)
    start_clicked = c1.button("Start", use_container_width=True, disabled=st.session_state.running)
    reset_clicked = c2.button("Reset", use_container_width=True)

    st.markdown("---")
    


if reset_clicked:
    st.session_state.running = False
    st.session_state.status = "Reset complete."
    st.session_state.history = deque(maxlen=500)
    st.session_state.timeline = deque(maxlen=60)
    st.session_state.agent_status = {"Affirmative": "Idle", "Negative": "Idle", "Judge": "Idle"}
    st.rerun()



arena_col, agents_col = st.columns([4.6, 1.4], gap="large")


arena_ph = arena_col.empty()
agents_ph = agents_col.empty()

with arena_ph.container():
    render_arena()

with agents_ph.container():
    render_agents_panel()



if start_clicked:
    st.session_state.running = True
    st.session_state.history = deque(maxlen=500)
    st.session_state.timeline = deque(maxlen=60)
    st.session_state.agent_status = {"Affirmative": "Idle", "Negative": "Idle", "Judge": "Idle"}

    push_status("Initializing…")
    push_timeline("Initializing…")

    
    with arena_ph.container():
        render_arena()
    with agents_ph.container():
        render_agents_panel()

    gen = build_orchestrator(st.session_state.topic, st.session_state.rounds, st.session_state.enable_rag)

    try:
        for event in gen:
            etype = event.get("type")

            if etype == "stage":
                nm = event.get("name", "")
                push_status(f"Stage: {nm}")
                push_timeline(f"Stage: {nm}")
                st.session_state.agent_status["Affirmative"] = "Listening"
                st.session_state.agent_status["Negative"] = "Listening"
                st.session_state.agent_status["Judge"] = "Idle"

            elif etype == "status":
                txt = event.get("text", "")
                push_status(txt)
                push_timeline(txt)

            elif etype == "msg":
                name = event.get("agent", "Agent")
                role = event.get("role", "Role")
                text = event.get("text", "")
                push_message(name, role, text)

                # mark who spoke
                if name == "Affirmative":
                    st.session_state.agent_status["Affirmative"] = "Speaking"
                    st.session_state.agent_status["Negative"] = "Listening"
                elif name == "Negative":
                    st.session_state.agent_status["Negative"] = "Speaking"
                    st.session_state.agent_status["Affirmative"] = "Listening"
                elif name == "Judge":
                    st.session_state.agent_status["Judge"] = "Speaking"

            elif etype == "done":
                push_status("Debate completed ✅")
                push_timeline("Debate completed ✅")
                st.session_state.agent_status["Affirmative"] = "Done"
                st.session_state.agent_status["Negative"] = "Done"
                st.session_state.agent_status["Judge"] = "Done"

           
            arena_ph.empty()
            agents_ph.empty()
            with arena_ph.container():
                render_arena()
            with agents_ph.container():
                render_agents_panel()

            time.sleep(0.03)

    except Exception as e:
        push_status(f"Error: {e}")
        arena_ph.empty()
        agents_ph.empty()
        with arena_ph.container():
            render_arena()
        with agents_ph.container():
            render_agents_panel()

    st.session_state.running = False
    st.balloons()
