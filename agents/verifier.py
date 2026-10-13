
# agents/verifier.py

from crewai import Agent

def create_agent(llm, tools):
    return Agent(
        role="Citation Verification Specialist",
        goal="Cross-check citation metadata against bibliographic records.",
        backstory=(
            "You compare DOI, title, authors and publication year. "
            "You label missing and mismatched metadata honestly. "
            "Metadata verification does not prove claim support."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )

