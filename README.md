# UWEPD - MCP Learning Project

A comprehensive learning project exploring Model Context Protocol (MCP) integration with Azure AI Agents and Streamlit applications.

## Overview

This project demonstrates various implementations of MCP (Model Context Protocol) clients using Azure AI Agents and Streamlit interfaces. It includes multiple versions of both Streamlit web applications and Python MCP clients for learning and experimentation purposes.

## Project Structure

```
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── streamlit_mcpclientapp.py      # Main Streamlit MCP client application
├── client.py                      # Python MCP client implementation
├── client_python.py               # Advanced MCP client (version 3)
├── backup/                        # Backup versions of Streamlit apps
│   ├── streamlit_mcpclient.py     # Original Streamlit client
│   ├── streamlit_mcpclient2.py    # Version 2
│   ├── streamlit_mcpclient3.py    # Version 3
│   ├── streamlit_mcpclient4.py    # Version 4
│   ├── streamlit_mcpclient5.py    # Version 5
│   ├── streamlit_mcpclient6.py    # Version 6
│   └── streamlit_mcpclient7.py    # Version 7
└── labvenv/                       # Virtual environment (not tracked)
```

## Features

### Streamlit Applications
- **Interactive Chat Interface**: Web-based chat with Azure AI Agents
- **MCP Tool Integration**: Direct access to Microsoft Learn documentation
- **Conversation Memory**: Persistent conversation history
- **Real-time Logging**: Transparent view of MCP tool calls and agent steps
- **Multiple Versions**: Various iterations showing different approaches and features

### Python MCP Clients
- **Direct MCP Communication**: Pure Python implementations
- **Azure AI Integration**: Seamless connection with Azure AI services
- **Version Progression**: Multiple versions demonstrating different capabilities

## Prerequisites

- Python 3.8+
- Azure AI subscription and configured credentials
- Environment variables set up (see Configuration section)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/convergent-analytics-ai-01/UWEPD.git
   cd UWEPD
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv labvenv
   # On Windows:
   labvenv\Scripts\activate
   # On macOS/Linux:
   source labvenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
PROJECT_ENDPOINT=your_azure_ai_project_endpoint
MODEL_DEPLOYMENT_NAME=your_model_deployment_name
```

### Azure Configuration
Ensure you have Azure credentials configured through one of:
- Azure CLI (`az login`)
- Environment variables
- Managed identity (when deployed)

## Usage

### Running Streamlit Applications

**Main Application:**
```bash
streamlit run streamlit_mcpclientapp.py
```

**Backup Versions:**
```bash
streamlit run backup/streamlit_mcpclient.py
```

### Running Python MCP Clients

```bash
python client.py
# or
python client_v2.py
# or
python client_v3.py
```

## MCP Integration

This project uses the Microsoft Learn MCP server (`https://learn.microsoft.com/api/mcp`) which provides:

- **Documentation Search**: Query the latest Microsoft documentation
- **Real-time Information**: Access to current Microsoft Learn content
- **Tool Transparency**: Detailed logging of MCP tool calls and responses

### MCP Features Demonstrated

1. **Tool Configuration**: Setting up MCP tools with Azure AI Agents
2. **Approval Modes**: Different MCP tool approval configurations
3. **Error Handling**: Robust error management for MCP operations
4. **Logging & Transparency**: Comprehensive logging of MCP interactions

## Key Components

### Azure AI Agent Integration
- Uses `DefaultAzureCredential` for authentication
- Creates ephemeral agents for each conversation
- Integrates MCP tools into agent toolsets
- Handles agent lifecycle management

### Memory Management
- File-based conversation persistence
- Thread-based conversation tracking
- JSON storage for conversation history
- Automatic memory directory creation

### Error Handling
- Comprehensive error catching and reporting
- Graceful degradation when services unavailable
- User-friendly error messages in Streamlit UI

## Learning Objectives

This project helps understand:

1. **MCP Protocol**: How to integrate external tools via MCP
2. **Azure AI Agents**: Working with Azure's AI agent framework
3. **Streamlit Development**: Building interactive AI applications
4. **Authentication**: Azure credential management
5. **Error Handling**: Robust error management in AI applications
6. **Version Control**: Git workflow and project organization

## Development

### Adding New Features
1. Create a new branch: `git checkout -b feature-name`
2. Make your changes
3. Test thoroughly
4. Commit: `git commit -m "Add feature description"`
5. Push: `git push origin feature-name`
6. Create pull request

### Version Management
The project maintains multiple versions to demonstrate evolution of features:
- Each version builds upon previous learnings
- Backup folder contains historical implementations
- Main files represent current best practices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please refer to individual component licenses for specific usage rights.

## Support

For questions or issues:
1. Check existing issues in the GitHub repository
2. Create a new issue with detailed description
3. Include relevant error messages and environment details

## Acknowledgments

- Microsoft Learn MCP Server
- Azure AI Services
- Streamlit framework
- Python MCP community

---

*This project is part of ongoing learning and experimentation with Model Context Protocol and Azure AI technologies.*