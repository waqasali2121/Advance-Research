
# agents/searcher.py

from crewai import Agent

def create_agent(llm, tools):
    return Agent(
        role="Academic Literature Searcher",
        goal="Find relevant scholarly records with traceable metadata.",
        backstory=(
            "You search academic sources, preserve DOI and URL "
            "information, and never invent publications."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )

