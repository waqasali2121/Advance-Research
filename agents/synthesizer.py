
# agents/synthesizer.py

from crewai import Agent

def create_agent(llm, tools):
    return Agent(
        role="Literature Review Synthesizer",
        goal="Write a balanced synthesis grounded in retrieved evidence.",
        backstory=(
            "You compare findings, explain contradictions, identify "
            "research gaps and cite only supplied sources."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )

