import os
from app.api.schemas.refactoring_assistant_schema import (
    AnalysisOutput, 
    CodeInput,
    RefactoredCode,
    RefactoringOpportunities,
    RefactoringSuggestions
)
from crewai import (
    Crew,
    Task, 
    Agent
)
from textwrap import dedent
import json
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

    def analysis_agent(self):
        # Agent 1: Code Analysis Expert
        tools = [PythonREPLTool()]
        return Agent(
            role="Code Analysis Expert",
            backstory=dedent("""You are an expert in analyzing code to detect issues, bugs, code smells, and provide complexity metrics."""),
            goal=dedent("""Analyze the provided code and identify any issues, potential bugs, or code smells. Also, provide complexity metrics such as cyclomatic complexity, maintainability index, and technical debt."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def opportunity_agent(self):
        # Agent 2: Refactoring Opportunity Identifier
        tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
                 WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
                 Tool( # Wrap ArxivAPIWrapper as a Tool
                     name="Arxiv",
                     func=ArxivAPIWrapper().run,
                     description="A wrapper around Arxiv. Useful for when you need to access academic papers."
                 )]
        return Agent(
            role="Refactoring Opportunity Identifier",
            backstory=dedent("""You specialize in identifying specific opportunities for code refactoring based on analysis results."""),
            goal=dedent("""Identify specific refactoring opportunities based on the analysis output, relating them to the identified issues, and assign a priority level."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def suggestion_agent(self):
        # Agent 3: Refactoring Suggestions Expert
        tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
                 WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
                 Tool( # Wrap ArxivAPIWrapper as a Tool
                     name="Arxiv",
                     func=ArxivAPIWrapper().run,
                     description="A wrapper around Arxiv. Useful for when you need to access academic papers."
                 )]
        return Agent(
            role="Refactoring Suggestions Expert",
            backstory=dedent("""You provide detailed suggestions on how to implement the identified refactoring opportunities, including estimated effort and affected dependencies."""),
            goal=dedent("""Generate detailed suggestions for implementing the refactoring opportunities, including estimated effort in hours and any affected dependencies."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def refactoring_agent(self):
        # Agent 4: Code Refactoring Specialist
        tools = [PythonREPLTool()]
        return Agent(
            role="Code Refactoring Specialist",
            backstory=dedent("""You apply refactoring suggestions to the code and produce the refactored code along with a summary of changes made and any new dependencies."""),
            goal=dedent("""Apply the refactoring suggestions to the code and produce the refactored code along with a summary of the changes made and any new dependencies introduced."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

class CustomTasks:
    def __init__(self):
        pass

    def code_analysis_task(self, agent, code_input: CodeInput):
        analysis_output_schema = AnalysisOutput.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Analyze the following {code_input.language} code snippet and identify any issues, potential bugs, or code smells.
                Also, provide complexity metrics such as cyclomatic complexity, maintainability index, and technical debt.
                Provide your output in **JSON format** matching the **AnalysisOutput** schema.

                **Format**:
                ```json
                {analysis_output_schema}
                ```

                **Code**:
                ```{code_input.language}
                {code_input.code_snippet}
                ```

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The analysis output in JSON format matching the schema: {analysis_output_schema}",
        )

    def refactoring_opportunity_task(self, agent, code_input: CodeInput):
        opportunities_schema = RefactoringOpportunities.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Based on the analysis of the following code, identify specific refactoring opportunities.
                Relate each opportunity to potential issues and assign a priority level (High, Medium, Low).
                Provide your output in **JSON format** matching the **RefactoringOpportunities** schema.

                **Format**:
                ```json
                {opportunities_schema}
                ```

                **Code**:
                ```{code_input.language}
                {code_input.code_snippet}
                ```

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The refactoring opportunities in JSON format matching the schema: {opportunities_schema}",
        )

    def refactoring_suggestion_task(self, agent, code_input: CodeInput):
        refactoring_suggestions_schema = RefactoringSuggestions.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Generate detailed suggestions for implementing refactoring opportunities in the following code.
                Include estimated effort in hours for each suggestion.
                Provide your output in **JSON format** matching the **RefactoringSuggestions** schema.

                **Format**:
                ```json
                {refactoring_suggestions_schema}
                ```

                **Code**:
                ```{code_input.language}
                {code_input.code_snippet}
                ```

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The refactoring suggestions in JSON format matching the schema: {refactoring_suggestions_schema}",
        )

    def code_refactoring_task(self, agent, code_input: CodeInput):
        refactored_code_schema = RefactoredCode.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Apply refactoring suggestions to the following code.
                Ensure the code remains functional and follows best practices.
                Provide the refactored code and a summary of changes made in **JSON format** matching the **RefactoredCode** schema.

                **Format**:
                ```json
                {refactored_code_schema}
                ```

                **Original Code**:
                ```{code_input.language}
                {code_input.code_snippet}
                ```

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The refactored code in JSON format matching the schema: {refactored_code_schema}",
        )

class CodeRefactoringCrew:
    def __init__(self, code_snippet, language="python", context=None):
        self.code_input = CodeInput(code_snippet=code_snippet, language=language, context=context)
        self.agents = CustomAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Define agents
        analysis_agent = self.agents.analysis_agent()
        opportunity_agent = self.agents.opportunity_agent()
        suggestion_agent = self.agents.suggestion_agent()
        refactoring_agent = self.agents.refactoring_agent()

        # Define tasks
        analysis_task = self.tasks.code_analysis_task(analysis_agent, self.code_input)
        opportunity_task = self.tasks.refactoring_opportunity_task(opportunity_agent, self.code_input)
        suggestion_task = self.tasks.refactoring_suggestion_task(suggestion_agent, self.code_input)
        refactoring_task = self.tasks.code_refactoring_task(refactoring_agent, self.code_input)

        # Create the crew
        crew = Crew(
            agents=[
                analysis_agent,
                opportunity_agent,
                suggestion_agent,
                refactoring_agent,
            ],
            tasks=[
                analysis_task,
                opportunity_task,
                suggestion_task,
                refactoring_task,
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result
    
def run_refactoring_assistant_crew(args: CodeInput):
    crew = CodeRefactoringCrew(args.code_snippet, args.language, args.context)
    results = crew.run()
    return results