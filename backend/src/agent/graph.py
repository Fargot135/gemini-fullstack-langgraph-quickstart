import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.tools_and_schemas import SearchQueryList
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    answer_instructions,
)
from agent.utils import get_research_topic

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")


def get_llm(temperature=0.1):
    """Initialize Groq Llama 3.3 70B with configurable temperature.
    
    Args:
        temperature: Sampling temperature for the model (default: 0.1)
        
    Returns:
        ChatGroq: Configured Groq LLM instance
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY")
    )


def search_local_files(directory: str, query: str):
    """Search for markdown files containing query keywords with relevance scoring.
    
    Args:
        directory: Root directory to search
        query: Search query string
        
    Returns:
        List of tuples (file_path, content) sorted by relevance
    """
    results = []
    query_lower = query.lower()
    query_keywords = [kw.lower() for kw in query.split() if len(kw) > 1]
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_lower = content.lower()
                        file_lower = file.lower()
                        path_lower = file_path.lower()
                        
                        score = 0
                        
                        # High score for filename matches
                        for keyword in query_keywords:
                            if keyword in file_lower:
                                score += 50
                            if keyword in path_lower:
                                score += 30
                        
                        # Medium score for content matches
                        for keyword in query_keywords:
                            score += content_lower.count(keyword) * 5
                        
                        # Bonus for exact phrase match
                        if query_lower in content_lower:
                            score += 100
                        
                        if score > 0:
                            relative_path = os.path.relpath(file_path, directory)
                            results.append((relative_path, content, score))
                except Exception:
                    continue
    
    results.sort(key=lambda x: x[2], reverse=True)
    return [(path, content) for path, content, _ in results]


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Groq Llama 3.3 70B to create optimized search queries for research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = get_llm(temperature=1.0)
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    try:
        result = structured_llm.invoke(formatted_prompt)
        return {"search_query": result.query}
    except Exception:
        # Fallback: use the original question as the search query
        return {"search_query": [get_research_topic(state["messages"])]}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the research node.

    This is used to spawn n number of research nodes, one for each search query.
    
    Args:
        state: Current graph state containing the generated search queries
        
    Returns:
        List of Send objects, one for each search query to process in parallel
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs local file research.

    Searches local markdown files for relevant content based on the search query,
    then processes the results with Groq Llama 3.3 70B for analysis.

    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable, including local_dir setting

    Returns:
        Dictionary with state update, including sources_gathered and web_research_results
    """
    local_dir = config.get("configurable", {}).get("local_dir")
    
    if not local_dir:
        raise ValueError("local_dir must be specified in config")
    
    search_results = search_local_files(local_dir, state["search_query"])
    
    if not search_results:
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"No relevant files found for query: {state['search_query']}"],
        }
    
    search_context = "\n\n---FILE---\n\n".join([
        f"File: {path}\n\n{content[:3000]}" 
        for path, content in search_results[:3]
    ])
    
    llm = get_llm(temperature=0)
    
    groq_prompt = f"""Analyze documentation to answer: {state["search_query"]}

Documentation files:

{search_context}

Provide a concise answer with specific code examples and syntax details.
IMPORTANT: Do not generate any web URLs or links. Only reference the file names provided."""
    
    try:
        groq_response = llm.invoke(groq_prompt)
        content = groq_response.content
    except Exception as e:
        content = f"Error analyzing files: {str(e)}"
    
    sources_gathered = [
        {"value": path, "short_url": path} 
        for path, _ in search_results[:3]
    ]
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [content],
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations using Groq Llama 3.3 70B.

    Args:
        state: Current graph state containing the running summary and sources gathered
        config: Configuration for the runnable

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    formatted_prompt += "\n\nIMPORTANT: Do not generate any web URLs or external links in your response."

    llm = get_llm(temperature=0)
    result = llm.invoke(formatted_prompt)

    unique_sources = []
    seen = set()
    for source in state["sources_gathered"]:
        if source["value"] not in seen:
            seen.add(source["value"])
            unique_sources.append(source)

    sources_text = "\n\nSources:\n" + "\n".join([f"- {s['value']}" for s in unique_sources])
    final_content = result.content + sources_text

    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": unique_sources,
    }


builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "finalize_answer")
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="local-search-agent")