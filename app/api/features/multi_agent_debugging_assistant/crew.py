from app.api.schemas.multi_agent_debugging_assistant_schema import (
    AnalysisOutput, 
    CodeInput, 
    DebuggingPlan, 
    FixSuggestions,
    FixedCode
)
from crewai import (
    Agent,
    Task,
    Crew
    )
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import Tool

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT4Mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4o", temperature=0)

    def bug_finder_agent(self):
        # Agent 1: Bug Finder
        tools = [PythonREPLTool()]
        return Agent(
            role="Bug Finder",
            backstory=dedent("""You are an expert in identifying bugs in code. You can detect syntax errors, runtime errors, logical errors, and any unexpected behavior."""),
            goal=dedent("""Examine the provided code and identify any bugs, errors, or anomalies. Provide detailed information about each bug found."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def bug_analyzer_agent(self):
        # Agent 2: Bug Analyzer
        tools = [
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
            WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="A wrapper around Arxiv. Useful for accessing academic papers."
            )
        ]
        return Agent(
            role="Bug Analyzer",
            backstory=dedent("""You specialize in analyzing bugs to determine their root causes and potential fixes."""),
            goal=dedent("""Analyze the identified bugs, determine their root causes, and assess their impact on the overall code."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def fix_planner_agent(self):
        # Agent 3: Fix Planner
        tools = [
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
            WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
            Tool(
                name="Arxiv",
                func=ArxivAPIWrapper().run,
                description="A wrapper around Arxiv. Useful for accessing academic papers."
            )
        ]
        return Agent(
            role="Fix Planner",
            backstory=dedent("""You provide detailed plans on how to fix the identified bugs, including estimated effort and any affected dependencies."""),
            goal=dedent("""Develop a step-by-step plan to fix the bugs, including priorities, estimated time, and required resources."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def code_fixer_agent(self):
        # Agent 4: Code Fixer
        tools = [PythonREPLTool()]
        return Agent(
            role="Code Fixer",
            backstory=dedent("""You apply fixes to the code based on the debugging plan and produce the fixed code along with a summary of changes made."""),
            goal=dedent("""Apply the fixes to the code and produce the fixed code along with a summary of the changes made and any new dependencies introduced."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

class CustomTasks:
    def __init__(self):
        pass

    def bug_finding_task(self, agent, code_input: CodeInput):
        analysis_output_schema = AnalysisOutput.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Examine the following {code_input.language} code snippet and identify any bugs, errors, or anomalies.
                Provide detailed information about each bug found.
                Provide your output in **JSON format** matching the **AnalysisOutput** schema.

                **Format**:

    json
{analysis_output_schema}

                **Code**:

    {code_input.language}
    {code_input.code_snippet}

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The analysis output in JSON format matching the schema: {analysis_output_schema}",
        )

    def bug_analysis_task(self, agent, code_input: CodeInput):
        debugging_plan_schema = DebuggingPlan.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Based on the identified bugs in the code, analyze each bug to determine its root cause and assess its impact.
                Provide your analysis in **JSON format** matching the **DebuggingPlan** schema.

                **Format**:

    json
{debugging_plan_schema}

                **Code**:

    {code_input.language}
    {code_input.code_snippet}

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The debugging plan in JSON format matching the schema: {debugging_plan_schema}",
        )

    def fix_planning_task(self, agent, code_input: CodeInput):
        fix_suggestions_schema = FixSuggestions.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Develop detailed suggestions for fixing the identified bugs in the following code.
                Include estimated effort in hours for each suggestion.
                Provide your output in **JSON format** matching the **FixSuggestions** schema.

                **Format**:

    json
{fix_suggestions_schema}

                **Code**:

    {code_input.language}
    {code_input.code_snippet}

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The fix suggestions in JSON format matching the schema: {fix_suggestions_schema}",
        )

    def code_fixing_task(self, agent, code_input: CodeInput):
        fixed_code_schema = FixedCode.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Apply the fix suggestions to the following code.
                Ensure the code remains functional and free of the identified bugs.
                Provide the fixed code and a summary of changes made in **JSON format** matching the **FixedCode** schema.

                **Format**:

    json
{fixed_code_schema}

                **Original Code**:

    {code_input.language}
    {code_input.code_snippet}

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The fixed code in JSON format matching the schema: {fixed_code_schema}",
        )
    
class DebuggingAssistantCrew:
    def __init__(self, code_snippet, language="python", context=None):
        self.code_input = CodeInput(code_snippet=code_snippet, language=language, context=context)
        self.agents = CustomAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Define agents
        bug_finder_agent = self.agents.bug_finder_agent()
        bug_analyzer_agent = self.agents.bug_analyzer_agent()
        fix_planner_agent = self.agents.fix_planner_agent()
        code_fixer_agent = self.agents.code_fixer_agent()

        # Define tasks
        bug_finding_task = self.tasks.bug_finding_task(bug_finder_agent, self.code_input)
        bug_analysis_task = self.tasks.bug_analysis_task(bug_analyzer_agent, self.code_input)
        fix_planning_task = self.tasks.fix_planning_task(fix_planner_agent, self.code_input)
        code_fixing_task = self.tasks.code_fixing_task(code_fixer_agent, self.code_input)

        # Create the crew
        crew = Crew(
            agents=[
                bug_finder_agent,
                bug_analyzer_agent,
                fix_planner_agent,
                code_fixer_agent,
            ],
            tasks=[
                bug_finding_task,
                bug_analysis_task,
                fix_planning_task,
                code_fixing_task,
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result

def run_multi_agent_debugging_crew(args: CodeInput):
    crew = DebuggingAssistantCrew(args.code_snippet, args.language, args.context)
    results = crew.run()
    return results