# Blog Generation Project using LangGraph

## Overview
This project is an automated blog generation system built with Python, leveraging the LangGraph framework and LLMs (Large Language Models) for dynamic content creation. It guides users through a multi-stage workflow to collect requirements, generate drafts, collect feedback, and produce finalized blog content.

## Features
- Interactive workflow for blog creation
- Modular graph-based orchestration using LangGraph
- Integration with Google Gemini LLM for content generation
- Feedback collection and revision loop
- Streamlit UI for user interaction
- Logging and state management

## Project Structure
```
app.py
requirements.txt
src/
  Blog/
    main.py
    graph/
      graph_builder_blog.py
    llms/
      chatgptllm.py
      geminillm.py
      groqllm.py
    logging/
      logging_utils.py
    nodes/
      blog_generation_node.py
    state/
      state.py
    tools/
      search_tool.py
    ui/
      streamlit/
        display_result_blog.py
```

## Getting Started
### Prerequisites
- Python 3.12+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set up environment variables (see `.env`):
  - `LANGCHAIN_API_KEY` for LLM access

### Running the App
1. Start the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
2. Follow the prompts to enter blog requirements and feedback.

## How It Works
- The workflow is managed by a LangGraph state machine defined in `graph_builder_blog.py`.
- Nodes represent stages: user input, LLM call, synthesis, feedback, revision, and file generation.
- The system allows users to choose their preferred LLM for generating blog drafts: Google Gemini, Groq, or ChatGPT.
- Feedback is collected and used to revise drafts until approval.

## Customization
- Add or modify LLMs in `src/Blog/llms/`
- Change workflow logic in `src/Blog/graph/graph_builder_blog.py`
- Update UI in `src/Blog/ui/streamlit/display_result_blog.py`

## Logging
Logs are stored in `src/Blog/logging/logs/app.log` for debugging and audit.

## License
MIT

## Author
kamaleswarmohanta@outlook.com
