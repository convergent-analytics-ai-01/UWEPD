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

def _memory_path_for_thread(thread_id: str) -> Path:
    mem_dir = Path("memory")
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir / f"conversation_{thread_id}.json"

def _load_memory(mem_path: Path) -> dict:
    if mem_path.exists():
        try:
            with mem_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If the file got corrupted, start fresh
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
    """
    Returns latest assistant text (and prints it).
    """
    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
    for m in msgs:
        if m.role == "assistant" and m.text_messages:
            last_text = m.text_messages[-1].text.value
            print(f"\nASSISTANT:\n{last_text}\n" + "-" * 50)
            return last_text
    print("\nASSISTANT: <no text found>\n" + "-" * 50)
    return ""

def _log_run_steps(agents_client: AgentsClient, thread_id: str, run_id: str):
    run_steps = agents_client.run_steps.list(thread_id=thread_id, run_id=run_id)
    for step in run_steps:
        print(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])
        if tool_calls:
            print("  MCP Tool calls:")
            for call in tool_calls:
                print(f"    Tool Call ID: {call.get('id')}")
                print(f"    Type: {call.get('type')}")
                print(f"    Name: {call.get('name')}")
        print()

# ---------------------------
# Main
# ---------------------------

def main():
    # Load environment variables
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Sanity checks
    if not project_endpoint or not model_deployment:
        raise ValueError(
            "PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME must be set in your .env file."
        )

    # Connect to Agents service
    agents_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
    )

    # MCP server config
    mcp_server_url = "https://learn.microsoft.com/api/mcp"
    mcp_server_label = "mslearn"

    # MCP tool setup
    mcp_tool = McpTool(server_label=mcp_server_label, server_url=mcp_server_url)
    mcp_tool.set_approval_mode("never")

    toolset = ToolSet()
    toolset.add(mcp_tool)

    # Use client as a context manager for clean resource handling
    with agents_client:
        agent = None
        try:
            # Create the agent (system instructions)
            agent = agents_client.create_agent(
                model=model_deployment,
                name="my-mcp-agent",
                instructions=(
                    "You are a helpful AI Agent for Tony.\n"
                    "If you use the MCP, greet with 'Hello Tony' and state that you are using MCP.\n"
                    "You have access to an MCP server called `microsoft.docs.mcp` which lets you "
                    "search the latest Microsoft documentation. Use MCP tools when helpful."
                    "**Contextual Statement Requirement**: The agent must include a brief context statement whenever it provides an answer based on MCP."
                    "- Example: Whenever you utilize the MCP for a response, please precede your answer with 'I am using the MCP to fetch this information.'"
                ),
            )

            print("\nAgent and Thread Info")
            print(f"Created agent, ID: {agent.id}")
            print(f"MCP Server: {mcp_tool.server_label} at {mcp_tool.server_url}")

            # Create a conversation thread (this carries memory within the session)
            thread = agents_client.threads.create()
            print(f"Created thread, ID: {thread.id}")

            # Prepare file-backed memory for this thread
            mem_path = _memory_path_for_thread(thread.id)
            mem_data = _load_memory(mem_path)
            mem_data["thread_id"] = thread.id
            with mem_path.open("w", encoding="utf-8") as f:
                json.dump(mem_data, f, ensure_ascii=False, indent=2)

            print("\nType your questions. Type 'exit' to quit.")
            print("-" * 50)

            while True:
                try:
                    user_input = input("\nYOU: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nExiting chat.")
                    break

                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                # Create user message on the thread
                user_msg = agents_client.messages.create(
                    thread_id=thread.id, role="user", content=user_input
                )
                _append_memory(mem_path, role="user", text=user_input, message_id=user_msg.id)

                # Run the agent with access to the MCP toolset
                run = agents_client.runs.create_and_process(
                    thread_id=thread.id, agent_id=agent.id, toolset=toolset
                )
                print(f"\nRun ID: {run.id}")
                print(f"Run completed with status: {run.status}")
                if run.status == "failed":
                    print(f"Run failed: {run.last_error}")

                # Log steps + tool calls for transparency
                _log_run_steps(agents_client, thread.id, run.id)

                # Print the latest assistant reply and store to memory
                assistant_text = _print_latest_assistant(agents_client, thread.id)
                if assistant_text:
                    # We don't have the assistant message id here directly; fetch the most recent assistant message
                    msgs = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING)
                    last_assistant_id = None
                    for m in msgs:
                        if m.role == "assistant":
                            last_assistant_id = getattr(m, "id", None)
                            break
                    _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

            # END while

        finally:
            # Clean up: delete agent to avoid cluttering your project with test agents
            if agent is not None:
                try:
                    agents_client.delete_agent(agent.id)
                    print("\nDeleted agent")
                except Exception as e:
                    print(f"\nWarning: could not delete agent ({e})")

if __name__ == "__main__":
    main()
