import argparse

from dotenv import load_dotenv

from workflow.graph import MAX_ITERATIONS, build_graph

load_dotenv()


def run_research(question: str) -> dict:
    """Run the complete bounded research workflow and return its final state."""
    graph = build_graph()
    return graph.invoke({"user_question": question, "max_iterations": MAX_ITERATIONS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent research system")
    parser.add_argument("question", help="Research question to investigate")
    args = parser.parse_args()
    result = run_research(args.question)
    print(result["final_report"])
    print("\nSources:")
    for source in result.get("sources", []):
        print(f"- {source.title}: {source.url}")
