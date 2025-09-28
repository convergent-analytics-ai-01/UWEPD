import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Azure Agents + Identity
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import McpTool, ToolSet, ListSortOrder

# --- NEW: Set page to wide layout. Must be the first st command. ---
st.set_page_config(layout="wide")

# ---------------------------
# Helpers: simple file memory
# ---------------------------
def _memory_dir() -> Path:
    d = Path("memory")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _memory_path_for_thread(thread_id: str) -> Path:
    return _memory_dir() / f"conversation_{thread_id}.json"

def _load_memory(mem_path: Path) -> dict:
    if mem_path.exists():
        try:
            with mem_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"thread_id": None, "messages": []}

def _append_memory(mem_path: Path, role: str, text: str, message_id: str = None):
    data = _load_memory(mem_path)
    data.setdefault("messages", [])
    data["messages"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "text": text,
        "message_id": message_id
    })
    data["messages"] = data["messages"][-20:] # Keep last 10 pairs
    with mem_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _print_latest_assistant(agents_client: AgentsClient, thread_id: str) -> str:
    st.session_state.activity_log.info("...Retrieving assistant's response.")
    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
    for m in msgs:
        if m.role == "assistant" and m.text_messages:
            last_text = m.text_messages[-1].text.value
            st.session_state.activity_log.success("✅ Response received.")
            return last_text
    st.session_state.activity_log.warning("⚠️ No assistant response found.")
    return ""

def _log_run_steps(agents_client: AgentsClient, thread_id: str, run_id: str):
    st.session_state.activity_log.subheader("Run Steps & Tool Calls")
    run_steps = agents_client.run_steps.list(thread_id=thread_id, run_id=run_id)
    with st.session_state.activity_log.expander("Show Details", expanded=True):
        for step in run_steps:
            st.write(f"**Step ID:** `{step.id}`")
            step_details = step.get("step_details", {})
            tool_calls = step_details.get("tool_calls", [])
            if tool_calls:
                st.write("**MCP Tool Calls:**")
                for call in tool_calls:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"ID: {call.get('id', 'N/A')}")
                    with col2:
                        st.text(f"Type: {call.get('type', 'N/A')}")
                    st.text(f"Name: {call.get('name', 'N/A')}")

# --- Helper functions for thread management ---
def get_all_threads() -> dict:
    threads = {}
    for mem_file in sorted(_memory_dir().glob("conversation_*.json"), reverse=True):
        mem_data = _load_memory(mem_file)
        thread_id = mem_data.get("thread_id")
        if thread_id and mem_data.get("messages"):
            first_user_message = next((msg["text"] for msg in mem_data["messages"] if msg["role"] == "user"), "Chat")
            threads[thread_id] = first_user_message
    return threads

def start_new_chat():
    st.session_state.messages = []
    st.session_state.selected_chat = "--- Start a new chat ---"
    if "thread_id" in st.session_state:
        del st.session_state["thread_id"]

def set_active_thread(thread_id: str):
    st.session_state.thread_id = thread_id
    mem_path = _memory_path_for_thread(thread_id)
    mem_data = _load_memory(mem_path)
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["text"]} for msg in mem_data.get("messages", [])
    ]

def delete_thread(thread_id: str):
    mem_path = _memory_path_for_thread(thread_id)
    try:
        if mem_path.exists():
            mem_path.unlink()
        if st.session_state.get("thread_id") == thread_id:
            start_new_chat()
    except Exception as e:
        st.error(f"Error deleting thread: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Azure AI Agent Chat 🤖")

# --- UPDATED: CSS for sidebar width AND dropdown text wrapping ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 450px !important;
        min-width: 450px !important;
    }
    /* Allow text wrapping in selectbox options */
    .st-bo li[role="option"] {
        white-space: normal;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.activity_log = st.sidebar
st.session_state.activity_log.header("Agent Activity Log 📜")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = "--- Start a new chat ---"

# --- UPDATED: Columns are swapped and resized ---
col1, col2 = st.columns([3, 1])

# --- Column 1: Main Chat Interface (Now on the left) ---
with col1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Column 2: Chat History (Now on the right with a dropdown) ---
with col2:
    st.subheader("Chat History")
    
    saved_threads = get_all_threads()
    # Create a list of options for the dropdown
    thread_options = list(saved_threads.values())
    
    # Define placeholder and options
    placeholder = "--- Select a conversation ---"
    options = [placeholder] + thread_options
    
    # Get the index of the currently selected chat to set the default value
    try:
        current_selection_index = options.index(st.session_state.get("selected_chat", placeholder))
    except ValueError:
        current_selection_index = 0

    selected_option = st.selectbox(
        "Load a previous chat:",
        options=options,
        index=current_selection_index,
        label_visibility="collapsed"
    )

    # Handle selection change
    if selected_option and selected_option != st.session_state.selected_chat:
        st.session_state.selected_chat = selected_option
        if selected_option == placeholder:
             start_new_chat()
        else:
            # Find the thread_id corresponding to the selected message
            thread_id_to_load = [tid for tid, msg in saved_threads.items() if msg == selected_option][0]
            set_active_thread(thread_id_to_load)
        st.rerun()

    # Add New Chat and Delete buttons
    st.button("➕ New Chat", on_click=start_new_chat, use_container_width=True)
    if st.session_state.get("thread_id"):
        st.button(
            "🗑️ Delete Current Chat",
            on_click=delete_thread,
            args=(st.session_state.thread_id,),
            use_container_width=True
        )


# --- Main Logic for Agent Interaction (inside Left Column) ---
with col1:
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment:
        st.error("PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME must be set in your .env file.")
    else:
        agents_client = AgentsClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
        mcp_tool = McpTool(server_label="mslearn", server_url="https://learn.microsoft.com/api/mcp")
        mcp_tool.set_approval_mode("never")
        toolset = ToolSet()
        toolset.add(mcp_tool)

        if prompt := st.chat_input("Ask your question:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if "thread_id" not in st.session_state:
                st.session_state.activity_log.info("Creating a new thread...")
                thread = agents_client.threads.create()
                st.session_state.thread_id = thread.id
                st.session_state.activity_log.success(f"✅ New thread: `{thread.id}`")

            thread_id = st.session_state.thread_id
            mem_path = _memory_path_for_thread(thread_id)
            
            # Agent Interaction Logic
            with st.spinner("Agent is processing..."):
                st.session_state.activity_log.info("Creating agent...")
                agent = agents_client.create_agent(
                    model=model_deployment, name="my-mcp-agent",
                    instructions=(
                        "You are a helpful AI Agent for Tony, specializing in Microsoft documentation.\n"
                        "Your **primary directive** is to use the `mslearn` MCP tool to answer any questions related to Microsoft products, services, technologies, or documentation.\n"
                        "You **must not** answer these questions from your own pre-existing knowledge. Always prioritize fetching the most current information from the MCP tool.\n"
                        "When the MCP tool is used, your response **must** begin with 'Hello Tony, I am using the MCP to fetch this information.'"
                    )
                )
                st.session_state.activity_log.success(f"✅ Agent created: `{agent.name}`")

                st.session_state.activity_log.info(f"➡️ Sending message...")
                user_msg = agents_client.messages.create(thread_id=thread_id, role="user", content=prompt)
                _append_memory(mem_path, role="user", text=prompt, message_id=user_msg.id)
                st.session_state.activity_log.success("✅ Message sent.")
                
                st.session_state.activity_log.info(f"🏃‍♂️ Starting run...")
                run = agents_client.runs.create_and_process(thread_id=thread_id, agent_id=agent.id, toolset=toolset)
                
                if run.status == "failed":
                    st.error(f"Run failed: {run.last_error}")
                    st.session_state.activity_log.error(f"❌ Run `{run.id}` failed.")
                else:
                    st.session_state.activity_log.success(f"✅ Run `{run.id}`: **{run.status}**")
                    _log_run_steps(agents_client, thread_id, run.id)

                    assistant_text = _print_latest_assistant(agents_client, thread_id)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                    
                    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
                    last_assistant_id = next((m.id for m in msgs if m.role == "assistant"), None)
                    _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

                st.session_state.activity_log.info(f"🗑️ Deleting agent `{agent.id}`...")
                try:
                    agents_client.delete_agent(agent.id)
                    st.session_state.activity_log.success("✅ Agent deleted.")
                except Exception as e:
                    st.warning(f"Could not delete agent: {e}")
                    st.session_state.activity_log.warning(f"⚠️ Could not delete agent: {e}")

            # Rerun to display the new messages immediately
            st.rerun()