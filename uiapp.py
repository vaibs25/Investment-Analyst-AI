import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from tavily import TavilyClient
import json
import gradio as gr

# Load environment variables
load_dotenv()

# Debug prints
print("Environment variables:")
print(f"GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Not Set'}")
print(f"TAVILY_API_KEY: {'Set' if os.getenv('TAVILY_API_KEY') else 'Not Set'}")

# Initialize Groq model and Tavily client
model = Groq(id="llama3-70b-8192")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

print("Initialized Groq model and Tavily client")

# Custom tool: Tavily search
def tavily_search(query, max_results=5):
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# -------------------------
# AGENTS SETUP (with token limits)
# -------------------------

research_agent = Agent(
    name="ResearchAgent",
    model=model,
    tools=[tavily_search],
    instructions=[
        "You are an investment research agent. Collect concise and relevant financial data using Tavily.",
        "Summarize findings clearly within 1000 tokens. Include only key updates on market trends, stock news, and fundamentals.",
        "Avoid long lists or deep history. Include only top 3 relevant sources and cite them briefly."
    ],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="FinanceAgent",
    model=model,
    instructions=[
        "You are a financial analyst. Summarize key metrics such as P/E, ROE, debt, and margins based on the research data.",
        "Keep the output short, clear, and under 1000 tokens. Explain only the most significant 3-5 metrics in simple terms.",
        "Avoid repetition or detailed background data unless critical."
    ],
    markdown=True
)

analysis_agent = Agent(
    name="AnalysisAgent",
    model=model,
    instructions=[
        "You are an investment analyst. Based on financial interpretation, provide a concise Buy, Hold, or Sell recommendation.",
        "Justify your answer with 2-3 solid points. Keep the answer under 1000 tokens and avoid excessive explanation.",
        "Summarize investment risks and opportunities briefly."
    ],
    markdown=True
)

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
# MAIN FUNCTION
# -------------------------

def investment_advisor_ai(user_prompt):
    progress = gr.Progress()
    progress(0, desc="Starting research...")
    research_output = research_agent.run(user_prompt)
    
    progress(0.33, desc="Analyzing financials...")
    finance_output = finance_agent.run(f"Based on this data: {research_output.content}")
    
    progress(0.66, desc="Making investment recommendation...")
    analysis_output = analysis_agent.run(f"Based on the financial interpretation: {finance_output.content}")
    
    progress(0.9, desc="Compiling final report...")
    final_report = editor_agent.run(f"""
    Research Summary:
    {research_output.content}

    Financial Interpretation:
    {finance_output.content}

    Investment Analysis:
    {analysis_output.content}
    """)
    progress(1.0, desc="Done!")
    return final_report.content

# -------------------------
# GRADIO UI
# -------------------------

css = """
.container {max-width: 900px; margin: auto;}
.markdown {white-space: pre-wrap; font-family: 'Segoe UI', sans-serif;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Investment Analyst AI (Multi-Agent System)")
    gr.Markdown("Ask about any *company* or *stock* to get a complete multi-agent investment analysis.")

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(label="🔍 Enter Stock/Company Query", placeholder="e.g., Analyze the stock potential of Tesla", lines=2)
            submit_button = gr.Button("Generate Report 🚀")
        with gr.Column(scale=5):
            output = gr.Markdown(label="📑 Investment Report")

    submit_button.click(fn=investment_advisor_ai, inputs=user_input, outputs=output)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()