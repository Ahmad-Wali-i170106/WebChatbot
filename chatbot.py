import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import lancedb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime, tool
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import HTMLSectionSplitter
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger

from crawler import UrlCrawler

load_dotenv()


@dataclass
class WebsiteContext:
    table_name: str


def bs4_extractor(html: str) -> str:
    """Simple HTML -> text extractor used by some loaders."""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_from_website(db: lancedb.DBConnection, url: str = "https://www.sunmarke.com"):
    """
    Crawl the provided URL, split into sections, embed and add to a LanceDB table.

    The table name is derived deterministically from the URL's netloc (hostname).
    If a table already exists with the same name it will be dropped and recreated.
    """
    table_name = urlparse(url).netloc.replace("www.", "").lower().replace(" ", "_")
    html_splitter = HTMLSectionSplitter(
        [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
    )

    # Crawl the website
    crawler = UrlCrawler(url)
    crawled_data = crawler.crawl(max_depth=3)

    # Convert crawled data to LangChain Documents
    docs = [
        Document(page_content=item.get("text", ""), metadata=item.get("metadata", {}))
        for item in crawled_data
    ]

    # Split the documents into HTML sections (if any)
    split_docs = html_splitter.split_documents(docs)

    # Generate embeddings using Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    texts = [doc.page_content for doc in split_docs]
    vectors = embeddings.embed_documents(texts)

    # Prepare rows for LanceDB. Flatten metadata to top-level columns.
    lance_rows = []
    for doc, vec in zip(split_docs, vectors):
        row = {"text": doc.page_content, "vector": vec}
        if isinstance(doc.metadata, dict):
            row.update(doc.metadata)
        lance_rows.append(row)

    # Drop table if it exists, then create new one with data
    if table_name in db.table_names():
        logger.info(f"Table {table_name} already exists - dropping previous table")
        db.drop_table(table_name)
    logger.info(f"Creating new table {table_name} with crawled website data")

    # Create table with inferred schema and data
    # DBConnection.create_table accepts name and data=... (lancedb will infer schema)
    db.create_table(table_name, data=lance_rows)

    # Return table name for caller convenience
    return table_name


def retrieve_context(
    query: str,
    max_results: int = 5,
    vector_store: Optional[LanceDB] = None,
    table_name: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    """
    Retrieve contextual documents from the vector_store.

    - If `table_name` is provided, it will be used deterministically to choose which
      LanceDB table to search.
    - If `vector_store` is None, a default local LanceDB connection is created.
    Returns: (joined_text, list_of_documents)
    """
    if vector_store is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        # default local DB path - caller can and should provide a LanceDB instance
        vector_store = LanceDB(
            uri="./data/lancedb-web-chatbot",
            embedding=embeddings,
            table_name="website-data",
            distance="l2",
        )

    # Determine which underlying LanceDB table to search.
    # The LanceDB wrapper's `similarity_search` accepts a `name` parameter which is
    # the table name to target. Use the provided table_name if given, otherwise use
    # the vector_store's configured table name.
    search_table = (
        table_name or getattr(vector_store, "table_name", None) or "website-data"
    )

    results = vector_store.similarity_search(query, k=max_results, name=search_table)

    text_results = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in results
    )

    return text_results, results


def create_rag_agent(vector_store: LanceDB):
    """
    Create a LangChain agent that deterministically injects retrieved context from a
    runtime-selected LanceDB table into the system prompt via middleware.

    The runtime `table_name` is provided by the application when invoking the agent,
    for example:
        invocation_config = {"configurable": {"thread_id": "1"}}
        agent.stream(user_messages, invocation_config, context=WebsiteContext(table_name="<website-data>", stream_mode="values")
    """
    # tools = [
    #     Tool.from_function(
    #         partial(retrieve_context, vector_store=vector_store),
    #         name="retrieve_context",
    #         description="Retrieve context from a LanceDB table (deterministic; app should pass table_name in invocation config).",
    #         response_format="content_and_artifact",
    #     )
    # ]

    @tool(
        response_format="content_and_artifact",
        description="Retrieve context from a LanceDB table (deterministic; app should pass table_name in invocation config).",
    )
    def retrieve_context_tool(
        query: str, runtime: ToolRuntime[WebsiteContext]
    ) -> Tuple[str, List[Document]]:
        # table_name = runtime.config.get("table_name", "website-data")
        table_name = runtime.context.table_name or "website-data"
        joined_text, docs = retrieve_context(
            query,
            max_results=5,
            vector_store=vector_store,
            table_name=table_name,
        )
        return joined_text, docs

    base_prompt = (
        "You are a helpful, polite, and concise AI assistant who helps users access and understand information "
        "from provided website context. Use the provided context to answer questions accurately and succinctly. "
        "Keep your responses professional and to the point."
    )

    agent = create_agent(
        model="google_genai:gemini-2.5-flash",
        tools=[retrieve_context_tool],
        system_prompt=base_prompt,
        checkpointer=InMemorySaver(),
        context_schema=WebsiteContext,
    )
    return agent


def create_rag_chain(vector_store: LanceDB):
    """
    Create an agent-like chain (no external tools) that deterministically injects retrieved
    context before each request. Behaves similarly to create_rag_agent with middleware.
    """
    from langchain.agents.middleware import dynamic_prompt

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        # Extract user query
        last_query = ""
        try:
            last_query = request.state["messages"][-1].text
        except Exception:
            last_query = ""
        table_name = request.runtime.context.table_name or "website-data"

        # Retrieve docs
        try:
            joined_text, _ = retrieve_context(
                last_query,
                max_results=5,
                vector_store=vector_store,
                table_name=table_name,
            )
        except Exception as exc:
            logger.exception("Error retrieving context for chain: %s", exc)
            joined_text = ""

        system_message = (
            f"Use the following context in your response:\n\n{joined_text}"
            if joined_text
            else ""
        )
        return system_message

    base_prompt = (
        "You are a helpful, polite, and concise AI assistant who helps users access and understand information "
        "from provided website context. Use the provided context to answer questions accurately and succinctly. "
        "Keep your responses professional and to the point."
    )

    agent = create_agent(
        "google_genai:gemini-2.5-flash",
        tools=[],
        system_prompt=base_prompt,
        middleware=[prompt_with_context],
        checkpointer=InMemorySaver(),
        context_schema=WebsiteContext,
    )
    return agent


if __name__ == "__main__":
    # Example usage:
    lancedb_conn = lancedb.connect("data/lancedb-web-chatbot")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vector_store = LanceDB(
        connection=lancedb_conn,
        embedding=embeddings,
        table_name="website-data",
        distance="l2",
    )

    # Crawl and create a table for the example site (deterministic table name derived from URL)
    created_table = load_from_website(lancedb_conn, url="https://www.funavry.com")
    logger.info(f"Created table: {created_table}")

    # Build agent
    agent = create_rag_agent(vector_store)

    while True:
        query = input("Please enter any query: ")
        if query.lower() in ["quit", "q", "exit"]:
            break

        for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "1"}},
            context=WebsiteContext(table_name="funavry.com"),
            stream_mode="messages",
        ):
            print(token)
            print(metadata)
