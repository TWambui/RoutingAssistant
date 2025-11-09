# AI Travel Decision Agent

## Demo Video

[![Watch the demo](https://youtu.be/tFObc_97joc)](https://youtu.be/YOUR_VIDEO_ID)

An AI-powered agent that helps determine whether you should leave for a location based on the origin and destination. The agent leverages mapping and AI tools to provide real-time travel insights.

## Features

- Predicts travel feasibility between two locations
- Integrates AI models for decision-making
- Visualizes routes and relevant travel information

## Installation

1. **Install `uv`** (assuming you already have Python installed):

   ```bash
   pip install uv
   ```

2. **Initialize a new project**:

   ```bash
   uv init .
   ```

3. **Add required dependencies**:

   ```bash
   uv add python-dotenv langgraph "langchain[anthropic]" ipykernel langchain-openai osmnx
   uv add scikit-learn matplotlib
   ```

4. **Configure environment variables**: Add your API keys and other secrets to a `.env` file in the project root.

## Usage

Run the main script:

```bash
uv run main_.py
```

Follow the prompts to input your origin and destination locations. The AI agent will analyze and provide a recommendation on whether to leave.

