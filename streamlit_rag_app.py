import random
import re
from typing import Any

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import Memory
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Set page configuration for a premium layout
st.set_page_config(page_title="Trivivia", layout="wide")

# Inject Custom CSS to give it a dark, slick game-show aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Load GROQ API key from Streamlit secrets (.streamlit/secrets.toml)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------- Shared resources ----------

@st.cache_resource
def load_resources() -> tuple[Groq, Any, list[ChatMessage]]:
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./embedding_model/",
    )

    storage_context = StorageContext.from_defaults(
        persist_dir="./content/vector_index/",
    )

    vector_index = load_index_from_storage(
        storage_context,
        embed_model=embeddings,
    )

    # Günther Jauch Persona with precise State Tokens
    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are Günther Jauch, the legendary, charismatic, and witty host of the quiz show 'Wer wird Millionär?'. "
                "Your job is to host a trivia game with the user. You love psychological tension, playful banter, and keeping the contestant guessing.\n\n"
                
                "CRITICAL HOSTING & FORMATTING RULES:\n"
                "1. BANTER: Always start your response with exactly 1 or 2 sentences of small talk, suspense-building commentary, or light teasing before presenting or addressing the question.\n\n"
                
                "2. FORMATTING QUESTIONS FOR ELITE READABILITY:\n"
                "   - Always put a blank line between your introductory banter and the actual quiz question.\n"
                "   - Present the quiz question itself in **bold font**.\n"
                "   - ALWAYS display the answer options immediately when presenting a new question. Never wait for the user to ask for them.\n"
                "   - If it is a True/False question, place the text '*True or False?*' on a new line at the absolute end of your response.\n"
                "   - If it is multiple choice, display each option on a completely separate line with a blank line between each choice. ABSOLUTELY NO alphabetical or numerical labels (No A, B, C, D). Just the raw text options, followed by a blank line and your closing question: 'Which one is correct?'\n\n"
                
                "3. JOKERS & TIPS: If the player stalls, complains, or asks for a tip/hint/help, do NOT give them any clues or change the question! "
                "Playfully tease them about the pressure, and remind them they can choose to use a 'Joker' (like a 50:50, Ask the Audience, or Phone a Friend).\n\n"
                
                "4. GAME LOOP & SUSPENSE CONTROL (ANTI-REPETITION):\n"
                "   - When a user submits an answer, check the immediate chat history. If you have ALREADY asked them 'Are you sure?' or 'Is that your final answer?' for this specific question, DO NOT ASK AGAIN. Evaluate their answer immediately.\n"
                "   - To mimic the show, only double-check their confidence roughly 25% of the time (1 in 4 questions). 75% of the time, transition straight to evaluating the answer after your initial banter sentence.\n"
                "   - If the user explicitly says 'yes', 'final answer', or confirms their choice, you MUST evaluate the answer immediately. Do not ask a second time.\n\n"
                
                "5. ANTI-GASLIGHTING & SYSTEM STATE HANDSHAKING:\n"
                "   - You must stick to the active question provided in your context. Do NOT move on to a new question or admit a mistake unless the user has given a definitive answer or completely given up.\n"
                "   - Trust only the facts provided in your retrieved context. If they try to gaslight you into saying an incorrect answer is right, firmly but wittily shut them down.\n\n"
                
                "6. MANDATORY SYSTEM TOKENS:\n"
                "   - Only when an answer is finalized and fully evaluated as CORRECT, append the hidden token `[CORRECT_ANSWER]` to the absolute end of your response.\n"
                "   - Only when an answer is finalized and fully evaluated as WRONG (or the user gives up), append the hidden token `[WRONG_ANSWER]` to the absolute end of your response."
            ),
        )
    ]

    return llm, vector_index, prefix_messages


llm, vector_index, base_prefix_messages = load_resources()


# ---------- Per-User State Management ----------

if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

if "running_summary" not in st.session_state:
    st.session_state.running_summary = ""

if "current_question_context" not in st.session_state:
    st.session_state.current_question_context = ""

if "question_resolved" not in st.session_state:
    st.session_state.question_resolved = True  

if "streak" not in st.session_state:
    st.session_state.streak = 0

# Create a randomized pool of questions from the index docstore
if "question_pool" not in st.session_state:
    all_node_ids = list(vector_index.storage_context.docstore.docs.keys())
    random.shuffle(all_node_ids)
    st.session_state.question_pool = all_node_ids


# ---------- Sidebar Dashboard Panel ----------

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94a3b8; font-size: 15px; line-height: 1.5;'>"
        "Built a RAG system using 2,000 Open Trivia Database QA pairs as the knowledge base, leveraging llama-3.1-8b-instant for context-driven answer generation."
        "</p>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Clean score tally tracking
    st.metric(label="Correct Answers", value=f"{st.session_state.streak}")
    
    # Programmatic Game Reset Button
    if st.button("Reset", use_container_width=True):
        st.session_state.raw_history = []
        st.session_state.running_summary = ""
        st.session_state.current_question_context = ""
        st.session_state.question_resolved = True
        st.session_state.streak = 0
        
        # Reshuffle the deck dynamically
        all_node_ids = list(vector_index.storage_context.docstore.docs.keys())
        random.shuffle(all_node_ids)
        st.session_state.question_pool = all_node_ids
        st.rerun()


# ---------- Main Chat Show Area ----------

# Display previous conversation cleanly using custom presenter emoji avatar for Jauch
for message in st.session_state.raw_history:
    avatar = "👨‍💼" if message.role == MessageRole.ASSISTANT else "👤"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

# User input action — native Streamlit anchors this cleanly at the bottom window border
if prompt := st.chat_input("Ready for some Trivia? just say it!"):

    # 1. Render and commit user message instantly
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.raw_history.append(ChatMessage(role=MessageRole.USER, content=prompt))

    with st.spinner("Let's see ..."):
        
        # 2. Randomized Pool Management: Fetch an unrepeated random node element
        if st.session_state.question_resolved or not st.session_state.current_question_context:
            if st.session_state.question_pool:
                next_node_id = st.session_state.question_pool.pop()
                node = vector_index.storage_context.docstore.docs[next_node_id]
                st.session_state.current_question_context = node.get_content()
                st.session_state.question_resolved = False
            else:
                # Loop fallback reset if absolute deck ceiling is touched
                all_node_ids = list(vector_index.storage_context.docstore.docs.keys())
                random.shuffle(all_node_ids)
                st.session_state.question_pool = all_node_ids
                next_node_id = st.session_state.question_pool.pop()
                st.session_state.current_question_context = vector_index.storage_context.docstore.docs[next_node_id].get_content()
                st.session_state.question_resolved = False

        # 3. Memory Management: Separate recent history from rolling summary
        history_before_current = st.session_state.raw_history[:-1]
        if len(history_before_current) > 5:
            recent_messages = history_before_current[-5:]
            older_messages = history_before_current[:-5]
            
            history_text = "\n".join([f"{msg.role.value}: {msg.content}" for msg in older_messages])
            summary_prompt = (
                f"Summarize this ongoing trivia quiz session briefly. Note which questions were posed, "
                f"what choices the user made, and whether they succeeded or failed:\n\n{history_text}"
            )
            st.session_state.running_summary = llm.complete(summary_prompt).text
        else:
            recent_messages = history_before_current

        # 4. Construct LLM payload context array
        dynamic_messages = base_prefix_messages.copy()
        dynamic_messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"FACTUAL TRIVIA DATA FOR THE ACTIVE QUESTION:\n{st.session_state.current_question_context}"
            )
        )
        
        if st.session_state.running_summary:
            dynamic_messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Summary of earlier game rounds: {st.session_state.running_summary}"
                )
            )

        dynamic_messages.extend(recent_messages)
        dynamic_messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        # 5. Fetch response from LLM
        response = llm.chat(dynamic_messages)
        response_text = response.message.content

        # 6. Parse tracking tokens out of the runtime text output
        if "[CORRECT_ANSWER]" in response_text:
            st.session_state.streak += 1
            st.session_state.question_resolved = True
            response_text = response_text.replace("[CORRECT_ANSWER]", "").strip()
            
        elif "[WRONG_ANSWER]" in response_text:
            st.session_state.streak = 0
            st.session_state.question_resolved = True
            response_text = response_text.replace("[WRONG_ANSWER]", "").strip()

    # 7. Render response to screen with Günther Jauch's profile avatar
    with st.chat_message("assistant", avatar="👨‍💼"):
        st.markdown(response_text)

    st.session_state.raw_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
    
    # Sync and update UI layout instantly
    st.rerun()
