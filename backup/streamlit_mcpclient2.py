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
    
    # Using an expander in the sidebar for collapsibility
    with st.session_state.activity_log.expander("Show Details", expanded=True):
        for step in run_steps:
            st.write(f"**Step ID:** `{step.id}`")
            # st.write(f"**Status:** `{step.status}`")
            
            step_details = step.get("step_details", {})
            tool_calls = step_details.get("tool_calls", [])
            
            if tool_calls:
                st.write("**MCP Tool Calls:**")
                for call in tool_calls:
                    # Use columns for a cleaner layout
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"ID: {call.get('id', 'N/A')}")
                    with col2:
                        st.text(f"Type: {call.get('type', 'N/A')}")
                    st.text(f"Name: {call.get('name', 'N/A')}")
            # st.divider()


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Azure AI Agent Chat 🤖")
st.write("Ask a question and get a response from your Azure AI Agent (with MCP Tool).")

# --- Inject custom CSS for a wider sidebar ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 400px !important;
        min-width: 400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar for Activity Logging ---
st.session_state.activity_log = st.sidebar
st.session_state.activity_log.header("Agent Activity Log 📜")

# Load environment variables
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

if not project_endpoint or not model_deployment:
    st.error("PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME must be set in your .env file.")
else:
    # Connect to Agents service
    agents_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
    )

    # MCP server config + tool
    mcp_server_url = "https://learn.microsoft.com/api/mcp"
    mcp_server_label = "mslearn"
    mcp_tool = McpTool(server_label=mcp_server_label, server_url=mcp_server_url)
    mcp_tool.set_approval_mode("never")
    toolset = ToolSet()
    toolset.add(mcp_tool)

    # Use Streamlit session state for thread
    if "thread_id" not in st.session_state:
        st.session_state.activity_log.info("No thread found. Creating a new thread...")
        thread = agents_client.threads.create()
        st.session_state["thread_id"] = thread.id
        st.session_state.activity_log.success(f"✅ New thread created: `{thread.id}`")
        mem_path = _memory_path_for_thread(thread.id)
        mem_data = _load_memory(mem_path)
        mem_data["thread_id"] = thread.id
        with mem_path.open("w", encoding="utf-8") as f:
            json.dump(mem_data, f, ensure_ascii=False, indent=2)
    else:
        mem_path = _memory_path_for_thread(st.session_state["thread_id"])
        st.session_state.activity_log.info(f"Using existing thread: `{st.session_state['thread_id']}`")


    # Create agent each run (will be deleted after response)
    st.session_state.activity_log.info("Creating agent for this session...")
    agent = agents_client.create_agent(
        model=model_deployment,
        name="my-mcp-agent",
        instructions=(
            "You are a helpful AI Agent for Tony, specializing in Microsoft documentation.\n"
            "Your **primary directive** is to use the `mslearn` MCP tool to answer any questions related to Microsoft products, services, technologies, or documentation.\n"
            "You **must not** answer these questions from your own pre-existing knowledge unless you check the MCP tool first and no answer is available. Always prioritize fetching the most current information from the MCP tool.\n"
            "When the MCP tool is used, your response **must** begin with 'Hello Tony, I am using the MCP to fetch this information.'"
        ),
    )
    st.session_state.activity_log.success(f"✅ Agent created: `{agent.name}` (`{agent.id}`)")


    user_input = st.text_input("Ask your question:")
    submit = st.button("Send")

    if submit and user_input:
        # Post user message
        st.session_state.activity_log.info(f"➡️ Sending user message to thread...")
        user_msg = agents_client.messages.create(
            thread_id=st.session_state["thread_id"], role="user", content=user_input
        )
        _append_memory(mem_path, role="user", text=user_input, message_id=user_msg.id)
        st.session_state.activity_log.success("✅ Message sent.")

        # Run the agent with the MCP toolset
        st.session_state.activity_log.info(f"🏃‍♂️ Starting run with agent...")
        run = agents_client.runs.create_and_process(
            thread_id=st.session_state["thread_id"], agent_id=agent.id, toolset=toolset
        )
        
        if run.status == "failed":
            st.error(f"Run failed: {run.last_error}")
            st.session_state.activity_log.error(f"❌ Run `{run.id}` failed.")
        else:
             st.session_state.activity_log.success(f"✅ Run `{run.id}` completed with status: **{run.status}**")


        # Transparency: log steps & tool calls
        _log_run_steps(agents_client, st.session_state["thread_id"], run.id)

        # Print & store latest assistant reply
        assistant_text = _print_latest_assistant(agents_client, st.session_state["thread_id"])
        if assistant_text:
            st.success(f"**Agent Response:**\n\n{assistant_text}")
            # Grab the latest assistant message id for bookkeeping
            msgs = agents_client.messages.list(thread_id=st.session_state["thread_id"], order=ListSortOrder.DESCENDING)
            last_assistant_id = None
            for m in msgs:
                if m.role == "assistant":
                    last_assistant_id = getattr(m, "id", None)
                    break
            _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

        # Delete agent after response
        st.session_state.activity_log.info(f"🗑️ Deleting agent `{agent.id}`...")
        try:
            agents_client.delete_agent(agent.id)
            st.session_state.activity_log.success("✅ Agent deleted.")
        except Exception as e:
            st.warning(f"Could not delete agent: {e}")
            st.session_state.activity_log.warning(f"⚠️ Could not delete agent: {e}")