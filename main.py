from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import requests
import sqlite3
import os


os.environ["OPENAI_API_KEY"] = ""
os.environ["WOLFRAM_ALPHA_APPID"] = ""


@tool
def wikipedia_search(query: str) -> str:
    """Vyhledá informace na Wikipedii (MCP REST)."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json().get("extract", "Nenalezeno.")
    return "Chyba při vyhledávání."

@tool
def sql_query(query: str) -> str:
    """Provede SQL dotaz na SQLite databázi."""
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return str(result)
    except Exception as e:
        return f"Chyba SQL: {str(e)}"

@tool
def wolfram_query(question: str) -> str:
    """Dotaz na Wolfram Alpha API přes MCP REST."""
    params = {
        "input": question,
        "appid": os.environ["WOLFRAM_ALPHA_APPID"],
        "format": "plaintext"
    }
    r = requests.get("https://api.wolframalpha.com/v1/result", params=params)
    if r.status_code == 200:
        return r.text
    return "Chyba při dotazu na Wolfram Alpha."

TOOLS = {
    "Wikipedia": wikipedia_search,
    "SQL": sql_query,
    "Wolfram": wolfram_query
}


planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    query: str
    plan: str
    results: List[str]
    answer: str


def planner_node(state: AgentState) -> AgentState:
    prompt = f"""
Jsi plánovací modul agenta. Máš dotaz uživatele: "{state['query']}".
Vytvoř jasný krokový plán, které nástroje použít a v jakém pořadí.
Dostupné nástroje: {list(TOOLS.keys())}
Formát: seznam kroků.
"""
    plan = planner_llm.invoke(prompt).content
    state["plan"] = plan
    return state


def executor_node(state: AgentState) -> AgentState:
    results = []
    for line in state["plan"].split("\n"):
        for tool_name, tool_func in TOOLS.items():
            if tool_name.lower() in line.lower():
                query_part = line.replace(tool_name, "").strip(": -")
                result = tool_func.run(query_part) if hasattr(tool_func, 'run') else tool_func(query_part)
                results.append(f"{tool_name} → {result}")
    state["results"] = results
    return state


def finish_node(state: AgentState) -> AgentState:
    final_prompt = f"""
Na základě těchto výsledků odpověz na původní dotaz uživatele.
Dotaz: {state['query']}
Výsledky kroků:
{chr(10).join(state['results'])}
"""
    answer = executor_llm.invoke(final_prompt).content
    state["answer"] = answer
    return state


workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("finish", finish_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "finish")
workflow.add_edge("finish", END)

graph = workflow.compile()


if __name__ == "__main__":
    print("💬 MCP LangGraph Agent (napiš 'exit' pro konec)")
    while True:
        user_query = input("\n❓ Zadej dotaz: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Ukončuji.")
            break
        initial_state: AgentState = {
            "query": user_query,
            "plan": "",
            "results": [],
            "answer": ""
        }
        result = graph.invoke(initial_state)
        print(f"\n✅ Odpověď:\n{result['answer']}")
