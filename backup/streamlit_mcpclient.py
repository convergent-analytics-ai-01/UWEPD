
import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
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
        "ts": datetime.utcnow().isoformat() + "Z",
        "role": role,
        "text": text,
        "message_id": message_id
    })
    with mem_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _print_latest_assistant(agents_client: AgentsClient, thread_id: str) -> str:
    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
    for m in msgs:
        if m.role == "assistant" and m.text_messages:
            last_text = m.text_messages[-1].text.value
            return last_text
    return ""

def _log_run_steps(agents_client: AgentsClient, thread_id: str, run_id: str):
    run_steps = agents_client.run_steps.list(thread_id=thread_id, run_id=run_id)
    logs = []
    for step in run_steps:
        logs.append(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])
        if tool_calls:
            logs.append("  MCP Tool calls:")
            for call in tool_calls:
                logs.append(f"    Tool Call ID: {call.get('id')}")
                logs.append(f"    Type: {call.get('type')}")
                logs.append(f"    Name: {call.get('name')}")
    return "\n".join(logs)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Azure AI Agent Chat")
st.write("Ask a question and get a response from your Azure AI Agent (with MCP Tool).")

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
        thread = agents_client.threads.create()
        st.session_state["thread_id"] = thread.id
        mem_path = _memory_path_for_thread(thread.id)
        mem_data = _load_memory(mem_path)
        mem_data["thread_id"] = thread.id
        with mem_path.open("w", encoding="utf-8") as f:
            json.dump(mem_data, f, ensure_ascii=False, indent=2)
    else:
        mem_path = _memory_path_for_thread(st.session_state["thread_id"])

    # Create agent each run (will be deleted after response)
    agent = agents_client.create_agent(
        model=model_deployment,
        name="my-mcp-agent",
        instructions=(
            "You are a helpful AI Agent for Tony.\n"
            "If you use the MCP, greet with 'Hello Tony' and state that you are using MCP.\n"
            "You have access to an MCP server called `microsoft.docs.mcp` which lets you "
            "search the latest Microsoft documentation. Use MCP tools when helpful.\n"
            "**Contextual Statement Requirement**: When an answer is based on MCP, begin with "
            "'I am using the MCP to fetch this information.'"
        ),
    )

    user_input = st.text_input("Ask your question:")
    submit = st.button("Send")

    if submit and user_input:
        # Post user message
        user_msg = agents_client.messages.create(
            thread_id=st.session_state["thread_id"], role="user", content=user_input
        )
        _append_memory(mem_path, role="user", text=user_input, message_id=user_msg.id)

        # Run the agent with the MCP toolset
        run = agents_client.runs.create_and_process(
            thread_id=st.session_state["thread_id"], agent_id=agent.id, toolset=toolset
        )
        st.write(f"Run ID: {run.id}")
        st.write(f"Run completed with status: {run.status}")
        if run.status == "failed":
            st.error(f"Run failed: {run.last_error}")

        # Transparency: log steps & tool calls
        logs = _log_run_steps(agents_client, st.session_state["thread_id"], run.id)
        if logs:
            st.expander("Run Steps & MCP Tool Calls").write(logs)

        # Print & store latest assistant reply
        assistant_text = _print_latest_assistant(agents_client, st.session_state["thread_id"])
        if assistant_text:
            st.success(assistant_text)
            # Grab the latest assistant message id for bookkeeping
            msgs = agents_client.messages.list(thread_id=st.session_state["thread_id"], order=ListSortOrder.DESCENDING)
            last_assistant_id = None
            for m in msgs:
                if m.role == "assistant":
                    last_assistant_id = getattr(m, "id", None)
                    break
            _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

        # Delete agent after response
        try:
            agents_client.delete_agent(agent.id)
        except Exception as e:
            st.warning(f"Could not delete agent: {e}")
