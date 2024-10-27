from app.api.schemas.llm_app_development_assistant_schema import ApplicationIdea, DevelopmentOutput
from crewai import (
    Agent,
    Task,
    Crew
)
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivAPIWrapper
from langchain.tools import Tool

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT4Mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4o", temperature=0)

    def feasibility_agent(self):
        # Agent 1: Feasibility Analyst
        tools = [
            TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
            ),
            Tool(
                name="Wikipedia",
                func=WikipediaAPIWrapper().run,
                description="Access Wikipedia articles for information."
            ),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="Access academic papers from Arxiv."
            ),
        ]
        return Agent(
            role="Feasibility Analyst",
            backstory=dedent("""You are an expert in assessing the feasibility of software projects, especially those involving LLMs. You utilize tools like Wikipedia and Arxiv to gather necessary information."""),
            goal=dedent("""Analyze the user's application idea and determine the feasibility of developing the desired LLM application. Provide detailed reasons and recommendations, using the provided tools to support your analysis."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def design_agent(self):
        # Agent 2: Solution Architect
        tools = [
            TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
            ),
            Tool(
                name="Wikipedia",
                func=WikipediaAPIWrapper().run,
                description="Access Wikipedia articles for information."
            ),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="Access academic papers from Arxiv."
            ),
        ]
        return Agent(
            role="Solution Architect",
            backstory=dedent("""You specialize in designing architectures for applications that leverage LLMs. You frequently consult resources like Wikipedia and Arxiv for the latest design patterns and technologies."""),
            goal=dedent("""Provide a detailed design architecture for the LLM application, including components, data flow, and integrations. Use the provided tools to enhance your design recommendations."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def implementation_agent(self):
        # Agent 3: Implementation Planner
        tools = [
            TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
            ),
            Tool(
                name="Wikipedia",
                func=WikipediaAPIWrapper().run,
                description="Access Wikipedia articles for information."
            ),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="Access academic papers from Arxiv."
            ),
        ]
        return Agent(
            role="Implementation Planner",
            backstory=dedent("""You provide detailed implementation plans for software projects involving LLMs. You utilize tools like Wikipedia and Arxiv to inform your planning."""),
            goal=dedent("""Create an implementation plan for the LLM application, including timeline, cost estimation, and resource requirements. Use the provided tools to inform your plan."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def output_agent(self):
        # Agent 4: Development Advisor
        tools = [
            TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
            ),
            Tool(
                name="Wikipedia",
                func=WikipediaAPIWrapper().run,
                description="Access Wikipedia articles for information."
            ),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="Access academic papers from Arxiv."
            ),
        ]
        return Agent(
            role="Development Advisor",
            backstory=dedent("""You combine all the information and provide a comprehensive development output to the user. You leverage tools like Wikipedia and Arxiv to ensure your advice is well-informed."""),
            goal=dedent("""Using the initial application idea, provide a comprehensive development output, including feasibility, design architecture, recommended tools, implementation plan, and other relevant details, matching the DevelopmentOutput schema. Use the provided tools to enhance your recommendations."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

class CustomTasks:
    def __init__(self):
        pass

    def feasibility_task(self, agent, application_idea: ApplicationIdea):
        return Task(
            description=dedent(f"""
                Analyze the following application idea and determine the feasibility of developing the desired LLM application.
                Provide detailed reasons and recommendations.
                **Use the tools provided to support your analysis.**

                **Application Idea**:

    {application_idea.model_dump_json(indent=2)}

            """),
            agent=agent,
            expected_output="A feasibility analysis with detailed reasons and recommendations.",
        )

    def design_task(self, agent, application_idea: ApplicationIdea):
        return Task(
            description=dedent(f"""
                Provide a detailed design architecture for the following application idea.
                Include components, data flow, and integrations.
                **Use the tools provided to enhance your design recommendations.**

                **Application Idea**:

    {application_idea.model_dump_json(indent=2)}

            """),
            agent=agent,
            expected_output="A design architecture including components, data flow, and integrations.",
        )

    def implementation_task(self, agent, application_idea: ApplicationIdea):
        return Task(
            description=dedent(f"""
                Create an implementation plan for the following application idea.
                Include timeline, cost estimation, and resource requirements.
                **Use the tools provided to inform your plan.**

                **Application Idea**:

    {application_idea.model_dump_json(indent=2)}

            """),
            agent=agent,
            expected_output="An implementation plan including timeline, cost estimation, and resource requirements.",
        )

    def development_output_task(self, agent, application_idea: ApplicationIdea):
        development_output_schema = DevelopmentOutput.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Using the initial application idea, provide a comprehensive development output.
                Include feasibility, design architecture, recommended tools, implementation plan, and other relevant details.
                Provide your output in **JSON format** matching the **DevelopmentOutput** schema.
                **Use the tools provided to ensure your advice is well-informed.**

                **Format**:

        json
{development_output_schema}

                **Application Idea**:

    {application_idea.model_dump_json(indent=2)}

            """),
            agent=agent,
            expected_output=f"The development output in JSON format matching the schema: {development_output_schema}",
        )

class LLMDevelopmentAssistantCrew:
    def __init__(self, project_name, description):
        self.application_idea = ApplicationIdea(
            project_name=project_name,
            description=description
        )
        self.agents = CustomAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Define agents
        feasibility_agent = self.agents.feasibility_agent()
        design_agent = self.agents.design_agent()
        implementation_agent = self.agents.implementation_agent()
        output_agent = self.agents.output_agent()

        # Define tasks
        feasibility_task = self.tasks.feasibility_task(feasibility_agent, self.application_idea)
        design_task = self.tasks.design_task(design_agent, self.application_idea)
        implementation_task = self.tasks.implementation_task(implementation_agent, self.application_idea)
        development_output_task = self.tasks.development_output_task(output_agent, self.application_idea)

        # Create the crew
        crew = Crew(
            agents=[
                feasibility_agent,
                design_agent,
                implementation_agent,
                output_agent,
            ],
            tasks=[
                feasibility_task,
                design_task,
                implementation_task,
                development_output_task,
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result
    
def run_llm_development_assistant_crew(args: ApplicationIdea):
    crew = LLMDevelopmentAssistantCrew(args.project_name, args.description)
    results = crew.run()
    return results