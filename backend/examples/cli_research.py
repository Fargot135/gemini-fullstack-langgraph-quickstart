import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory to search for markdown files",
    )
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=1,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--reasoning-model",
        default="llama-3.3-70b-versatile",
        help="Model for reasoning and final answer (default: llama-3.3-70b-versatile)",
    )
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "reasoning_model": args.reasoning_model,
    }

    config = {"configurable": {"local_dir": args.dir}}

    result = graph.invoke(state, config=config)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)
        
    # Print sources if available
    sources = result.get("sources_gathered", [])
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source.get('value', 'N/A')}")


if __name__ == "__main__":
    main()