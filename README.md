# Web Chatbot

A small project that demonstrates crawling a website, storing website content in LanceDB, creating embeddings (via Google's Gemini embeddings), and exposing a simple conversational interface over that website data. The project includes:
The project includes:

- a crawler (`crawler.py`) that extracts text and metadata from a site,
- an interactive agent-based chatbot example (`chatbot.py`),
- a Streamlit UI (`streamlit_ui.py`) to ingest sites and chat interactively.

This README gives beginner-friendly instructions for installing and running the app locally.

---

## Prerequisites

- Python 3.14 or later (project `pyproject.toml` declares `requires-python = ">=3.14"`).
- Basic familiarity with creating and activating a virtual environment.
- A Gemini (Google Generative AI) API key to enable embeddings and model calls (the code refers to this as `GOOGLE_API_KEY`). Without this key, features that call Google GenAI may fail or be disabled.

Optional (depends on your environment):
- A working installation of `lancedb` (the project dependencies include `lancedb`).
- Network access from the machine to the websites you want to crawl and to the Google Generative AI endpoints.

---

## Quick start — Setup

1. Open a terminal and change into the project directory:

   ```
   cd web_chatbot
   ```

2. Create and activate a Python virtual environment (recommended):

   - On macOS / Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. Install project dependencies:

The project includes a `pyproject.toml`. From inside the `web_chatbot` directory you can install the package (and its dependencies) in editable mode:

```
pip install -e .
```

If you use the `uv` package manager, it can consume the `pyproject.toml` for dependency management — see the official docs at https://docs.astral.sh/uv/ for full details. A minimal `uv` workflow for this project is:

```
uv init .    # Create the UV folder
uv sync      # Install all the requirements
```

To run the Streamlit app using `uv`:

```
uv run streamlit run streamlit_ui.py
```

If `pip install -e .` fails because a build/backend is not configured for your environment, you can use `uv` as shown above or install dependencies manually from `pyproject.toml` (or by installing the main packages listed there, e.g. `streamlit`, `lancedb`, `langchain`, `langchain-google-genai`, `sentence-transformers`, etc.). `pip install -e .` remains a valid fallback for environments that support it.

4. (Optional) Create a `.env` file in `web_chatbot/` or export environment variables directly. The Streamlit UI and other modules look for `GOOGLE_API_KEY` for embeddings/model calls:

   - Example `.env`:
     ```
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```

   - Or export in your shell:
     ```
     export GOOGLE_API_KEY="your_gemini_api_key_here"
     ```

   Note: The Streamlit UI also provides a sidebar field to paste the key at runtime (it stores it for the session only).

---

## Running the app

There are a few different ways to run and explore the project.

1. Run the Streamlit User Interface (recommended for beginners)

   The Streamlit app lets you ingest websites (crawl + index) and chat with an agent that uses the site context.

   From inside the `web_chatbot` directory:

   ```
   streamlit run streamlit_ui.py
   ```

   - Open the URL printed by Streamlit (default: http://localhost:8501).
   - In the sidebar: enter/paste your Gemini API key (or leave blank but model calls will fail if missing), type a website URL, and click "Ingest and select site".
   - After ingestion completes you can ask questions in the chat area. Ingesting large sites may take time.

2. Run the example CLI / scraper

A small demo scraper script that previously lived in `main.py` has been removed from the repository. For CLI-style interaction and examples, use the interactive console chatbot in `chatbot.py` (see the `__main__` example in that file), or use the Streamlit UI described above to ingest and interact with websites.

3. Run the interactive console chatbot

   `chatbot.py` contains a more feature-rich example that:
   - crawls websites with the included `UrlCrawler`,
   - splits content and creates embeddings,
   - writes to a LanceDB table,
   - and runs a simple streaming agent loop in the console.

   Run it from the `web_chatbot` directory:

   ```
   python chatbot.py
   ```

   The script's `__main__` block shows example usage that will:
   - connect to a local LanceDB at `data/lancedb-web-chatbot`,
   - crawl a configured site,
   - build a vector store and agent, and
   - allow you to type queries in a loop.

   Note: This script depends on the Gemini embeddings (`GoogleGenerativeAIEmbeddings`) and model access. Make sure `GOOGLE_API_KEY` is set.

---

## Data storage

- Local LanceDB files are stored under the `data/` directory by default (for example, `data/lancedb-web-chatbot`).

---

## How ingestion works (high level)

1. Crawling: `UrlCrawler` follows same-domain links, extracts visible text and some metadata, and respects robots.txt (it will attempt to parse and follow crawl delay).
2. Splitting: HTML sections are split into manageable chunks (see `HTMLSectionSplitter` usage).
3. Embeddings: Gemini embeddings (`gemini-embedding-001`) are used to convert text chunks into vectors.
4. Storage: Chunks + vectors are stored in LanceDB tables (table names are derived from site hostnames).
5. Querying: The agent retrieves similar documents from LanceDB at runtime and injects them into the prompt for the model.

---

## Environment variables & configuration

- `GOOGLE_API_KEY` — required for Google Generative AI embeddings and model calls (Gemini). You can either set this in your environment or paste it into the Streamlit sidebar.
- The code uses local on-disk LanceDB connections by default (no additional config required for local use).

---

## Troubleshooting

- Missing packages / import errors: ensure you've activated the virtual environment and run `pip install -e .` (or installed the dependencies listed in `pyproject.toml`).
- No API key provided: model and embedding calls will fail; set `GOOGLE_API_KEY` or provide it in the Streamlit UI.
- Crawl failures / blocked pages: some sites may block scraping with bot protection. Check the crawler logs and user-agent settings.
- LanceDB errors: ensure `lancedb` is installed and you have file system permissions to create files under `data/` or the working directory.
- Streamlit port conflicts: if port 8501 is in use, pass `--server.port <PORT>` to the `streamlit run` command.

---

## Notes, limitations, and safety

- This repository demonstrates prototype code for building a website-specific chatbot. It is not hardened for production usage.
- Respect websites' `robots.txt` and terms of service when crawling.
- The quality and correctness of answers depend on the upstream model and the quality of crawled content.
- Consider rate limits, quotas, and costs when using Gemini/Google GenAI APIs.

---

## Where to look in the repo

- `crawler.py` — robust site crawler with robots.txt handling and text extraction.
- `chatbot.py` — the more complete ingest → embed → agent example and a simple interactive console loop.
- `streamlit_ui.py` — Streamlit-based UI for ingesting sites and chatting interactively.
- Project utilities and LanceDB-related helpers are referenced within the code (see `chatbot.py` and the Streamlit app for examples of how LanceDB is used).
