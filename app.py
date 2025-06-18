import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from tavily import TavilyClient
import json
from phi.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

# Initialize the Groq LLM and Tavily client
model = Groq(id="llama3-70b-8192")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Custom tool function to perform Tavily search
def tavily_search(query, max_results=5):
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# -------------------------
# AGENTS SETUP (with token limits)
# -------------------------

# Agent 1: Research Agent
research_agent = Agent(
    name="ResearchAgent",
    model=model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    tools=[tavily_search],
    instructions=[
        "You are an investment research agent. Collect concise and relevant financial data using Tavily.",
        "Summarize findings clearly within 1000 tokens. Include only key updates on market trends, stock news, and fundamentals.",
        "Avoid long lists or deep history. Include only top 3 relevant sources and cite them briefly."
    ],
    show_tool_calls=True,
    markdown=True
)

# Agent 2: Finance Agent
finance_agent = Agent(
    name="FinanceAgent",
    model=model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=[
        "You are a financial analyst. Summarize key metrics such as P/E, ROE, debt, and margins based on the research data.",
        "Keep the output short, clear, and under 1000 tokens. Explain only the most significant 3-5 metrics in simple terms.",
        "Avoid repetition or detailed background data unless critical."
    ],
    markdown=True
)

# Agent 3: Analysis Agent
analysis_agent = Agent(
    name="AnalysisAgent",
    model=model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=[
        "You are an investment analyst. Based on financial interpretation, provide a concise Buy, Hold, or Sell recommendation.",
        "Justify your answer with 2-3 solid points. Keep the answer under 1000 tokens and avoid excessive explanation.",
        "Summarize investment risks and opportunities briefly."
    ],
    markdown=True
)

# Agent 4: Editor Agent
editor_agent = Agent(
    name="EditorAgent",
    model=model,
    instructions=[
        "You are a financial editor. Combine all agent outputs into a single professional summary.",
        "Ensure total report stays under 1500 tokens. Use headings, markdown formatting, and avoid repetition.",
        "Keep the language clear, engaging, and suitable for a decision-making investor."
    ],
    markdown=True
)

# -------------------------
# MULTI-STAGE FUNCTION
# -------------------------

def investment_advisor_ai(user_prompt):
    print("\n🔍 Step 1: Research Agent Running...")
    research_output = research_agent.run(user_prompt)

    print("\n📊 Step 2: Finance Agent Running...")
    finance_output = finance_agent.run(f"Based on this data: {research_output.content}")

    print("\n📈 Step 3: Analysis Agent Running...")
    analysis_output = analysis_agent.run(f"Based on the financial interpretation: {finance_output.content}")

    print("\n📝 Step 4: Editor Agent Compiling Final Report...")
    final_report = editor_agent.run(f"""
    Research Summary:
    {research_output.content}

    Financial Interpretation:
    {finance_output.content}

    Investment Analysis:
    {analysis_output.content}
    """)

    return final_report.content

# -------------------------
# CLI LOOP
# -------------------------

if __name__ == "__main__":
    print("💼 Welcome to Investment Analyst AI (Tavily Edition)")
    print("Ask about any company or stock to get a full analysis. Type 'exit' to quit.\n")

    while True:
        user_input = input("📈 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting... Happy Investing!")
            break
        report = investment_advisor_ai(user_input)
        print("\n📑 Final Investment Report:\n")
        print(report)