import os
from typing import Optional
from urllib.parse import urlparse

import lancedb
import streamlit as st
from langchain_community.vectorstores import LanceDB
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chatbot import WebsiteContext, create_rag_agent, load_from_website

st.set_page_config(page_title="Website Chatbot", page_icon="🤖")
st.title("Website Chatbot 🤖")


# -----------------------
# Utilities
# -----------------------
def table_name_from_url(url: str) -> str:
    """
    Deterministic table name derivation that matches chatbot.load_from_website().
    """
    if not url:
        return "website-data"
    parsed = urlparse(url if "://" in url else f"https://{url}")
    netloc = parsed.netloc or parsed.path
    return netloc.replace("www.", "").lower().replace(" ", "_").strip("/ ")


def stream_response(messages: list):
    """
    Stream the agent response for the given prompt.
    """

    for token, metadata in agent.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": "1"}},
        context=context,
        stream_mode="messages",
    ):
        # print(token, "\n\n")
        if (
            metadata["langgraph_node"] == "model"
            and len(token.content_blocks) > 0
            and token.content_blocks[-1]["type"] == "text"
        ):
            yield token.content_blocks[-1]["text"]


# -----------------------
# Cached resources (tied to the Gemini API key so that switching keys rebuilds embeddings/vectorstore/agent)
# -----------------------
@st.cache_resource(show_spinner=True)
def get_lancedb_connection():
    """
    Create or reuse an on-disk LanceDB connection.
    """
    # Path is deterministic for the application
    return lancedb.connect("data/lancedb-web-chatbot")


@st.cache_resource(show_spinner=True)
def get_embeddings(gemini_api_key: Optional[str]):
    """
    Create embeddings client. Cache is keyed by gemini_api_key so changing the key recreates embeddings.
    """
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    # Use the GoogleGenerativeAIEmbeddings wrapper (Gemini embeddings)
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# @st.cache_resource(show_spinner=True)
def get_vector_store(conn, embeddings):
    """
    Create a LanceDB-backed vector store wrapper used by the agent.
    The cache is keyed by gemini_api_key so it will be rebuilt if the key changes.
    """
    # Use the same on-disk path as examples in chatbot.py
    return LanceDB(
        connection=conn,
        embedding=embeddings,
        # table_name="website-data",
        distance="l2",
    )


# @st.cache_resource(show_spinner=True)
def get_agent_for_key(gemini_api_key: Optional[str], vector_store):
    """
    Create an agent wired to the provided vector_store. Cache keyed by gemini_api_key.
    """
    # Ensure env var is available for any downstream model creation
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    return create_rag_agent(vector_store)


# Build base resources (these are cached and will be recreated if the gemini key changes)
lancedb_conn = get_lancedb_connection()

# -----------------------
# Sidebar: Gemini API key and Ingest controls
# -----------------------
with st.sidebar:
    st.header("Settings")

    # Gemini (Google GenAI) API Key - keep in session state for the running session
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = os.environ.get("GOOGLE_API_KEY", "")

    gemini_key = st.text_input(
        "Gemini (Google Generative) API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        help="Enter your Gemini API key (used for embeddings and model calls). "
        "This will be stored only for the running Streamlit session.",
    )
    if gemini_key:
        st.session_state["gemini_api_key"] = gemini_key
        # Also set env var so underlying client libraries pick it up if they consult env.
        os.environ["GOOGLE_API_KEY"] = gemini_key
    else:
        st.warning(
            "Please enter your Gemini API key to enable embeddings and model calls."
        )
    st.markdown("---")
    st.header("Ingest / Select Website")

    # Known sites persisted in session state
    if "known_sites" not in st.session_state:
        st.session_state["known_sites"] = [
            f"https://{tab_name}" for tab_name in lancedb_conn.table_names()
        ]

    known_sites = st.session_state["known_sites"]

    # Selection: choose a known site or create a new one
    options = ["-- Enter new URL --"] + known_sites
    site_choice = st.selectbox(
        "Choose a previously ingested site or enter a new one:", options
    )

    new_url_input = ""
    if site_choice == "-- Enter new URL --":
        new_url_input = st.text_input(
            "New website URL to ingest",
            value=st.session_state.get("last_entered_url", "https://"),
            help="Enter the base URL of the website you want to chat about (e.g. https://example.com).",
        )
        ingest_label = "Ingest and select site"
    else:
        new_url_input = site_choice
        ingest_label = "Reload / Re-ingest selected site"

    ingest_btn = st.button(ingest_label)

    st.markdown(
        "Change the URL and press the Ingest button to crawl and index that website. "
        "Indexing may take some time for large sites."
    )


embeddings = get_embeddings(st.session_state.get("gemini_api_key"))
vector_store = get_vector_store(lancedb_conn, embeddings)
agent = get_agent_for_key(st.session_state.get("gemini_api_key"), vector_store)

# -----------------------
# Handle ingestion action (create/refresh per-site table)
# -----------------------
if ingest_btn:
    url_to_ingest = (new_url_input or "").strip()
    if not url_to_ingest or url_to_ingest == "https://":
        st.sidebar.error("Please enter a valid URL to ingest.")
    else:
        st.session_state["last_entered_url"] = url_to_ingest
        table_name = table_name_from_url(url_to_ingest)
        with st.spinner(f"Ingesting {url_to_ingest} -> table '{table_name}' ..."):
            # load_from_website will create/drop the LanceDB table for that site.
            created_table = load_from_website(lancedb_conn, url=url_to_ingest)
        # Add to known sites
        if url_to_ingest not in st.session_state["known_sites"]:
            st.session_state["known_sites"].append(url_to_ingest)
        # Set the current site selection
        st.session_state["current_site_url"] = url_to_ingest
        st.session_state["current_table_name"] = created_table
        # Initialize messages storage for the site if necessary
        if "messages_by_site" not in st.session_state:
            st.session_state["messages_by_site"] = {}
        if (
            st.session_state["current_table_name"]
            not in st.session_state["messages_by_site"]
        ):
            st.session_state["messages_by_site"][
                st.session_state["current_table_name"]
            ] = []
        # st.experimental_rerun()

# If user clicked the selection from dropdown but didn't press ingest, allow using the site
if site_choice != "-- Enter new URL --" and not ingest_btn:
    if st.sidebar.button("Use selected site"):
        st.session_state["current_site_url"] = site_choice
        st.session_state["current_table_name"] = table_name_from_url(site_choice)
        if "messages_by_site" not in st.session_state:
            st.session_state["messages_by_site"] = {}
        if (
            st.session_state["current_table_name"]
            not in st.session_state["messages_by_site"]
        ):
            st.session_state["messages_by_site"][
                st.session_state["current_table_name"]
            ] = []
        # st.experimental_rerun()

# -----------------------
# Determine current site/context
# -----------------------
current_url = st.session_state.get("current_site_url")
current_table_name = st.session_state.get("current_table_name")

if not current_url or not current_table_name:
    st.info(
        "No website selected. Use the sidebar to enter a URL and press 'Ingest and select site' or choose a known site and press 'Use selected site'."
    )
    st.stop()

# Ensure per-site messages structure
if "messages_by_site" not in st.session_state:
    st.session_state["messages_by_site"] = {}
if current_table_name not in st.session_state["messages_by_site"]:
    st.session_state["messages_by_site"][current_table_name] = []

st.subheader(f"Chatting about: {current_url} (table: {current_table_name})")

# Display chat history for current site
for msg in st.session_state["messages_by_site"][current_table_name]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and streaming response (scoped to current_table_name via WebsiteContext)
prompt = st.chat_input(f"Ask me anything about {current_url}...")
if prompt:
    # Save user message in site-specific history
    st.session_state["messages_by_site"][current_table_name].append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_agent = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages_by_site"][current_table_name]
    ]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        context = WebsiteContext(table_name=current_table_name)

        try:
            with st.spinner("Thinking..."):
                full_response = st.write_stream(stream_response(messages_for_agent))
        except Exception as exc:
            response_placeholder.error(f"Error getting response from agent: {exc}")
            full_response = f"Error: {exc}"

    # Append assistant reply to site history
    st.session_state["messages_by_site"][current_table_name].append(
        {"role": "assistant", "content": full_response}
    )

# Small footer controls
st.markdown("---")
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Clear chat for this site"):
        st.session_state["messages_by_site"][current_table_name] = []
        st.rerun()
with c2:
    st.caption(
        "Tip: Ingest each website you want to chat about. Then select it from the sidebar to focus the conversation on that site's data. Changing the Gemini API key will rebuild embeddings/agent for the session."
    )
