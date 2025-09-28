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

# Set page to wide layout. Must be the very first st command.
st.set_page_config(layout="wide")

# Load environment variables at the start of the script
load_dotenv()

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
    st.session_state.log_messages.append(("info", "...Retrieving assistant's response."))
    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
    for m in msgs:
        if m.role == "assistant" and m.text_messages:
            last_text = m.text_messages[-1].text.value
            st.session_state.log_messages.append(("success", "✅ Response received."))
            return last_text
    st.session_state.log_messages.append(("warning", "⚠️ No assistant response found."))
    return ""

def _log_run_steps(agents_client: AgentsClient, thread_id: str, run_id: str):
    st.session_state.log_messages.append(("subheader", "Run Steps & Tool Calls"))
    run_steps = agents_client.run_steps.list(thread_id=thread_id, run_id=run_id)
    step_details_for_log = []
    for step in run_steps:
        details = {"id": step.id, "tool_calls": []}
        step_details_obj = step.get("step_details", {})
        tool_calls = step_details_obj.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                details["tool_calls"].append({
                    "id": call.get('id', 'N/A'),
                    "type": call.get('type', 'N/A'),
                    "name": call.get('name', 'N/A')
                })
        step_details_for_log.append(details)
    st.session_state.log_messages.append(("run_steps", step_details_for_log))

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
    st.session_state.log_messages = []
    if "thread_id" in st.session_state:
        del st.session_state["thread_id"]

def set_active_thread(thread_id: str):
    st.session_state.thread_id = thread_id
    mem_path = _memory_path_for_thread(thread_id)
    mem_data = _load_memory(mem_path)
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["text"]} for msg in mem_data.get("messages", [])
    ]
    threads = get_all_threads()
    st.session_state.selected_chat = threads.get(thread_id, "--- Start a new chat ---")


def delete_thread(thread_id: str):
    mem_path = _memory_path_for_thread(thread_id)
    try:
        if mem_path.exists():
            mem_path.unlink()
        if st.session_state.get("thread_id") == thread_id:
            start_new_chat()
    except Exception as e:
        st.error(f"Error deleting thread: {e}")

def handle_chat_selection():
    selected_chat_message = st.session_state.selected_chat
    if selected_chat_message == "--- Start a new chat ---":
        start_new_chat()
        return
    
    saved_threads = get_all_threads()
    thread_id_to_load = [tid for tid, msg in saved_threads.items() if msg == selected_chat_message][0]
    set_active_thread(thread_id_to_load)

def delete_selected_chat_callback():
    selected_chat_message = st.session_state.selected_chat
    if selected_chat_message == "--- Start a new chat ---":
        st.toast("No chat selected to delete.")
        return

    saved_threads = get_all_threads()
    thread_id_to_delete = [tid for tid, msg in saved_threads.items() if msg == selected_chat_message][0]
    delete_thread(thread_id_to_delete)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("EPD 522 - Azure AI Agent Chat 🤖")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 450px !important;
        min-width: 450px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = "--- Start a new chat ---"
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

# --- Sidebar for Activity Logging ---
with st.sidebar:
    # --- NEW: Added a text logo/title at the top of the sidebar ---
    # st.markdown("# UW EPD 522")
    st.markdown('<h1 style="color: darkred;">Developing Agent MCP based Apps</h1>', unsafe_allow_html=True)
    st.divider() # Optional: adds a visual separator
    
    st.header("Agent Activity Log 📜")
    for log_type, content in st.session_state.log_messages:
        if log_type == "info":
            st.info(content)
        elif log_type == "success":
            st.success(content)
        elif log_type == "warning":
            st.warning(content)
        elif log_type == "error":
            st.error(content)
        elif log_type == "subheader":
            st.subheader(content)
        elif log_type == "run_steps":
            with st.expander("Show Details", expanded=True):
                for step in content:
                    st.write(f"**Step ID:** `{step['id']}`")
                    if step['tool_calls']:
                        st.write("**MCP Tool Calls:**")
                        for call in step['tool_calls']:
                             st.text(f"  ID: {call['id']}\n  Type: {call['type']}\n  Name: {call['name']}")

col1, col2 = st.columns([3, 1])

# --- Column 1: Main Chat Interface ---
with col1:
    if prompt := st.chat_input("Ask your question:"):
        st.session_state.log_messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        
        if "thread_id" not in st.session_state:
            st.session_state.log_messages.append(("info", "Creating a new thread..."))
            agents_client_for_thread = AgentsClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
            thread = agents_client_for_thread.threads.create()
            st.session_state.thread_id = thread.id
            st.session_state.log_messages.append(("success", f"✅ New thread: `{thread.id}`"))

        thread_id = st.session_state.thread_id
        mem_path = _memory_path_for_thread(thread_id)
        
        with st.spinner("Agent is processing..."):
            model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
            agents_client = AgentsClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
            mcp_tool = McpTool(server_label="mslearn", server_url="https://learn.microsoft.com/api/mcp")
            mcp_tool.set_approval_mode("never")
            toolset = ToolSet()
            toolset.add(mcp_tool)

            st.session_state.log_messages.append(("info", "Creating agent..."))
            agent = agents_client.create_agent(
                model=model_deployment, name="my-mcp-agent",
                instructions=(
                    "You are a helpful AI Agent for Tony, specializing in Microsoft documentation.\n"
                    "Your **primary directive** is to use the `mslearn` MCP tool to answer any questions related to Microsoft products, services, technologies, or documentation.\n"
                    "You **must not** answer these questions from your own pre-existing knowledge. Always prioritize fetching the most current information from the MCP tool.\n"
                    "When the MCP tool is used, your response **must** begin with 'Hello Tony, I am using the MCP to fetch this information.'"
                )
            )
            st.session_state.log_messages.append(("success", f"✅ Agent created: `{agent.name}`"))

            st.session_state.log_messages.append(("info", "➡️ Sending message..."))
            user_msg = agents_client.messages.create(thread_id=thread_id, role="user", content=prompt)
            _append_memory(mem_path, role="user", text=prompt, message_id=user_msg.id)
            st.session_state.log_messages.append(("success", "✅ Message sent."))
            
            st.session_state.log_messages.append(("info", "🏃‍♂️ Starting run..."))
            run = agents_client.runs.create_and_process(thread_id=thread_id, agent_id=agent.id, toolset=toolset)
            
            if run.status == "failed":
                st.error(f"Run failed: {run.last_error}")
                st.session_state.log_messages.append(("error", f"❌ Run `{run.id}` failed."))
            else:
                st.session_state.log_messages.append(("success", f"✅ Run `{run.id}`: **{run.status}**"))
                _log_run_steps(agents_client, thread_id, run.id)

                assistant_text = _print_latest_assistant(agents_client, thread_id)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                
                msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
                last_assistant_id = next((m.id for m in msgs if m.role == "assistant"), None)
                _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

            st.session_state.log_messages.append(("info", f"🗑️ Deleting agent `{agent.id}`..."))
            try:
                agents_client.delete_agent(agent.id)
                st.session_state.log_messages.append(("success", "✅ Agent deleted."))
            except Exception as e:
                st.warning(f"Could not delete agent: {e}")
                st.session_state.log_messages.append(("warning", f"⚠️ Could not delete agent: {e}"))
        
        st.rerun()

    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Column 2: Chat History ---
with col2:
    st.subheader("Chat History")
    
    saved_threads = get_all_threads()
    placeholder = "--- Start a new chat ---"
    options = [placeholder] + list(saved_threads.values())
    
    st.radio(
        "Saved Conversations:",
        options=options,
        key="selected_chat",
        on_change=handle_chat_selection,
        label_visibility="collapsed"
    )
    
    st.divider()

    st.button("➕ New Chat", on_click=start_new_chat, use_container_width=True)
    
    st.button(
        "🗑️ Delete Selected Chat",
        on_click=delete_selected_chat_callback,
        use_container_width=True,
        disabled=(st.session_state.selected_chat == placeholder)
    )