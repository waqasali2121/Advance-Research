
# agents/planner.py

from crewai import Agent

def create_agent(llm, tools):
    return Agent(
        role="Research Planning Specialist",
        goal="Create a focused and reproducible research strategy.",
        backstory=(
            "You define search concepts, inclusion criteria, "
            "exclusion criteria and research scope."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )
