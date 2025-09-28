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
    # Keep only the last 10 messages for the agent's context window
    data["messages"] = data["messages"][-20:] # 10 user + 10 assistant messages
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
    """Logs run steps and tool calls to the Streamlit sidebar."""
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

# --- NEW: Helper functions for thread management ---
def get_all_threads() -> dict:
    """Scans the memory directory and returns a dict of thread_id -> first_user_message."""
    threads = {}
    for mem_file in _memory_dir().glob("conversation_*.json"):
        mem_data = _load_memory(mem_file)
        thread_id = mem_data.get("thread_id")
        if thread_id and mem_data.get("messages"):
            first_user_message = next((msg["text"] for msg in mem_data["messages"] if msg["role"] == "user"), "Chat")
            threads[thread_id] = first_user_message
    return threads

def start_new_chat():
    """Resets the session state to start a new conversation."""
    st.session_state.messages = []
    if "thread_id" in st.session_state:
        del st.session_state["thread_id"]

def set_active_thread(thread_id: str):
    """Loads a previous conversation into the session state."""
    st.session_state.thread_id = thread_id
    mem_path = _memory_path_for_thread(thread_id)
    mem_data = _load_memory(mem_path)
    # Load messages, converting them to the format st.chat_message expects
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["text"]} for msg in mem_data.get("messages", [])
    ]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Azure AI Agent Chat 🤖")

# --- Inject custom CSS for a wider sidebar ---
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

# --- Sidebar for Activity Logging ---
st.session_state.activity_log = st.sidebar
st.session_state.activity_log.header("Agent Activity Log 📜")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main page layout with columns ---
col1, col2 = st.columns([1, 3])

# --- Column 1: Chat History ---
with col1:
    st.subheader("Chat History")
    st.button("➕ New Chat", on_click=start_new_chat, use_container_width=True)
    st.divider()
    
    saved_threads = get_all_threads()
    for thread_id, first_message in saved_threads.items():
        # Truncate the message for display
        display_message = (first_message[:30] + '...') if len(first_message) > 30 else first_message
        st.button(display_message, key=thread_id, on_click=set_active_thread, args=(thread_id,), use_container_width=True)

# --- Column 2: Main Chat Interface ---
with col2:
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load environment variables and set up clients
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

        # Get user input
        if prompt := st.chat_input("Ask your question:"):
            # Add user message to chat history and session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Check for active thread, create one if it doesn't exist
            if "thread_id" not in st.session_state:
                st.session_state.activity_log.info("No thread found. Creating a new thread...")
                thread = agents_client.threads.create()
                st.session_state["thread_id"] = thread.id
                st.session_state.activity_log.success(f"✅ New thread created: `{thread.id}`")

            thread_id = st.session_state["thread_id"]
            mem_path = _memory_path_for_thread(thread_id)
            
            # --- Agent Interaction ---
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                st.session_state.activity_log.info("Creating agent for this session...")
                agent = agents_client.create_agent(
                    model=model_deployment, name="my-mcp-agent",
                    instructions=(
                        "You are a helpful AI Agent for Tony, specializing in Microsoft documentation.\n"
                        "Your **primary directive** is to use the `mslearn` MCP tool to answer any questions related to Microsoft products, services, technologies, or documentation.\n"
                        "You **must not** answer these questions from your own pre-existing knowledge. Always prioritize fetching the most current information from the MCP tool.\n"
                        "When the MCP tool is used, your response **must** begin with 'Hello Tony, I am using the MCP to fetch this information.'"
                    )
                )
                st.session_state.activity_log.success(f"✅ Agent created: `{agent.name}` (`{agent.id}`)")

                # Post user message to the thread
                st.session_state.activity_log.info(f"➡️ Sending user message to thread...")
                user_msg = agents_client.messages.create(thread_id=thread_id, role="user", content=prompt)
                _append_memory(mem_path, role="user", text=prompt, message_id=user_msg.id)
                st.session_state.activity_log.success("✅ Message sent.")
                
                # Run the agent
                st.session_state.activity_log.info(f"🏃‍♂️ Starting run with agent...")
                run = agents_client.runs.create_and_process(thread_id=thread_id, agent_id=agent.id, toolset=toolset)
                
                if run.status == "failed":
                    st.error(f"Run failed: {run.last_error}")
                    st.session_state.activity_log.error(f"❌ Run `{run.id}` failed.")
                else:
                    st.session_state.activity_log.success(f"✅ Run `{run.id}` completed with status: **{run.status}**")
                    _log_run_steps(agents_client, thread_id, run.id)

                    # Get and display assistant response
                    assistant_text = _print_latest_assistant(agents_client, thread_id)
                    message_placeholder.markdown(assistant_text)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                    
                    # Save assistant response to memory
                    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
                    last_assistant_id = next((m.id for m in msgs if m.role == "assistant"), None)
                    _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

                # Delete agent
                st.session_state.activity_log.info(f"🗑️ Deleting agent `{agent.id}`...")
                try:
                    agents_client.delete_agent(agent.id)
                    st.session_state.activity_log.success("✅ Agent deleted.")
                except Exception as e:
                    st.warning(f"Could not delete agent: {e}")
                    st.session_state.activity_log.warning(f"⚠️ Could not delete agent: {e}")