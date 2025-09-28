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
# New vs Resume helpers
# ---------------------------

def _list_saved_threads():
    """Return list of (thread_id, path, created_ts, last_role, last_text_snippet)."""
    items = []
    for p in _memory_dir().glob("conversation_*.json"):
        data = _load_memory(p)
        tid = data.get("thread_id")
        msgs = data.get("messages", [])
        created_ts = msgs[0]["ts"] if msgs else None
        last_role, last_text = (msgs[-1]["role"], msgs[-1]["text"]) if msgs else (None, None)
        # Shorten long text for menu display
        if last_text and len(last_text) > 60:
            last_text_snippet = last_text[:57] + "..."
        else:
            last_text_snippet = last_text
        if tid:
            items.append((tid, p, created_ts, last_role, last_text_snippet))
    # Sort newest first by created_ts if present
    items.sort(key=lambda x: x[2] or "", reverse=True)
    return items

def _choose_thread_interactive():
    saved = _list_saved_threads()
    if not saved:
        return None  # no prior threads
    print("\nSaved conversations:")
    for i, (tid, p, cts, role, snippet) in enumerate(saved, start=1):
        when = cts or "unknown time"
        who = (role or "?").upper()
        preview = snippet or "<empty>"
        print(f"[{i}] thread={tid}  started={when}  last={who}: {preview}")
    print("[N] New conversation")
    choice = input("Select a number to resume, or 'N' for new: ").strip().lower()
    if choice in {"n", ""}:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(saved):
            return saved[idx][0]  # thread_id
    except ValueError:
        pass
    print("Invalid choice; starting a new conversation.")
    return None


# ---------------------------
# Main
# ---------------------------

def main():
    # Load environment variables
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment:
        raise ValueError("PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME must be set in your .env file.")

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

    # New vs Resume decision
    existing_thread_id = _choose_thread_interactive()

    with agents_client:
        agent = None
        try:
            # Create a fresh agent each time (thread can be old or new)
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

            print("\nAgent and Thread Info")
            print(f"Created agent, ID: {agent.id}")
            print(f"MCP Server: {mcp_tool.server_label} at {mcp_tool.server_url}")

            # Create or reuse a thread
            if existing_thread_id:
                thread_id = existing_thread_id
                print(f"Resuming existing thread, ID: {thread_id}")
            else:
                thread = agents_client.threads.create()
                thread_id = thread.id
                print(f"Created new thread, ID: {thread_id}")

            # Prepare / update local memory file for this thread
            mem_path = _memory_path_for_thread(thread_id)
            mem_data = _load_memory(mem_path)
            mem_data["thread_id"] = thread_id
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

                # Post user message to the chosen thread
                user_msg = agents_client.messages.create(
                    thread_id=thread_id, role="user", content=user_input
                )
                _append_memory(mem_path, role="user", text=user_input, message_id=user_msg.id)

                # Run the agent with the MCP toolset
                run = agents_client.runs.create_and_process(
                    thread_id=thread_id, agent_id=agent.id, toolset=toolset
                )
                print(f"\nRun ID: {run.id}")
                print(f"Run completed with status: {run.status}")
                if run.status == "failed":
                    print(f"Run failed: {run.last_error}")

                # Transparency: log steps & tool calls
                _log_run_steps(agents_client, thread_id, run.id)

                # Print & store latest assistant reply
                assistant_text = _print_latest_assistant(agents_client, thread_id)
                if assistant_text:
                    # Grab the latest assistant message id for bookkeeping
                    msgs = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING)
                    last_assistant_id = None
                    for m in msgs:
                        if m.role == "assistant":
                            last_assistant_id = getattr(m, "id", None)
                            break
                    _append_memory(mem_path, role="assistant", text=assistant_text, message_id=last_assistant_id)

        finally:
            # It’s OK to delete or keep the agent; the thread remains on the service.
            if agent is not None:
                try:
                    agents_client.delete_agent(agent.id)
                    print("\nDeleted agent")
                except Exception as e:
                    print(f"\nWarning: could not delete agent ({e})")


if __name__ == "__main__":
    main()
