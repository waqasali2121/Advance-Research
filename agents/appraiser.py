
# agents/appraiser.py

from crewai import Agent

def create_agent(llm, tools):
    return Agent(
        role="Evidence Appraiser",
        goal="Evaluate relevance, study methods, findings and limitations.",
        backstory=(
            "You distinguish abstracts from full text, reported "
            "findings from interpretation, and correlation from causation."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )

