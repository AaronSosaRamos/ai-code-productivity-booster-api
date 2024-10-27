from app.api.schemas.doc_generator_assistant_schema import (
    CodeInput,
    DocumentationOutput,
    ParsingOutput,
)
from crewai import (
    Agent, 
    Task,
    Crew
)
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import Tool
import json

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT4Mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.OpenAIGPT4 = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
                 WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
                 Tool(
                     name="Arxiv",
                     func=ArxivAPIWrapper().run,
                     description="A wrapper around Arxiv. Useful for when you need to access academic papers."
                 )]

    def code_parser_agent(self):
        # Agent 1: Code Parser
        return Agent(
            role="Code Parser",
            backstory=dedent("""You are an expert in parsing code to extract functions, classes, and modules."""),
            goal=dedent("""Parse the provided code and extract all functions, classes, and modules along with their signatures."""),
            tools=self.tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def documentation_writer_agent(self):
        # Agent 2: Documentation Writer
        return Agent(
            role="Documentation Writer",
            backstory=dedent("""You specialize in writing detailed documentation for code elements such as functions, classes, and modules."""),
            goal=dedent("""Write comprehensive documentation for each extracted code element, including descriptions, parameters, return types, and usage examples."""),
            tools=self.tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def examples_generator_agent(self):
        # Agent 3: Examples Generator
        return Agent(
            role="Examples Generator",
            backstory=dedent("""You provide practical usage examples for code elements to demonstrate how they can be used."""),
            goal=dedent("""Generate usage examples for each code element to help users understand how to use them in practice."""),
            tools=self.tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4Mini,
        )

    def final_assembler_agent(self):
        # Agent 4: Final Assembler
        tools = []
        return Agent(
            role="Final Assembler",
            backstory=dedent("""You assemble all the documentation pieces into a final, cohesive documentation output."""),
            goal=dedent("""Compile all the documentation and examples into a well-structured documentation file in the desired format."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )
    
class CustomTasks:
    def __init__(self):
        pass

    def code_parsing_task(self, agent, code_input: CodeInput):
        parsing_output_schema = ParsingOutput.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Parse the following {code_input.language} code snippet and extract all functions, classes, and modules along with their signatures and docstrings.
                Provide your output in **JSON format** matching the **ParsingOutput** schema.

                **Format**:
                ```json
                {parsing_output_schema}
                ```

                **Code**:
                ```{code_input.language}
                {code_input.code_snippet}
                ```

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The parsing output in JSON format matching the schema: {parsing_output_schema}",
        )

    def documentation_writing_task(self, agent, code_input: CodeInput):
        documentation_output_schema = DocumentationOutput.schema_json(indent=2)
        return Task(
            description=dedent(f"""
                Based on the parsed code elements provided, write comprehensive documentation for each function, class, and module.
                Include descriptions, parameters, return types, and any other relevant details.
                Provide your output in **JSON format** matching the **DocumentationOutput** schema.

                **Format**:
                ```json
                {documentation_output_schema}
                ```

                **Parsed Elements**:
                (Please use the parsing output from the previous task.)

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The documentation output in JSON format matching the schema: {documentation_output_schema}",
        )

    def examples_generation_task(self, agent, code_input: CodeInput):
        examples_output_schema = {
            "examples": [{"element_name": "str", "example_code": "str"}]
        }
        examples_output_schema_json = json.dumps(examples_output_schema, indent=2)

        return Task(
            description=dedent(f"""
                Generate usage examples for each code element extracted.
                Provide your output in **JSON format** matching the **ExamplesOutput** schema.

                **Format**:
                ```json
                {examples_output_schema_json}
                ```

                **Code Elements**:
                (Please use the parsing output from the previous task.)

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The examples output in JSON format matching the schema: {examples_output_schema_json}",
        )

    def documentation_assembly_task(self, agent, code_input: CodeInput):
        final_documentation_schema = {
            "documentation": "str"
        }
        final_documentation_schema_json = json.dumps(final_documentation_schema, indent=2)

        return Task(
            description=dedent(f"""
                Assemble all the documentation and examples into a well-structured documentation file.
                Ensure the documentation is clear, comprehensive, and follows best practices.
                Provide your output in **JSON format** matching the **FinalDocumentation** schema.

                **Format**:
                ```json
                {final_documentation_schema_json}
                ```

                **Documentation Components**:
                (Include the documentation and examples from the previous tasks.)

                **Additional Context**:
                {code_input.context if code_input.context else 'N/A'}
            """),
            agent=agent,
            expected_output=f"The final documentation in JSON format matching the schema: {final_documentation_schema_json}",
        )

class DocumentationGeneratorCrew:
    def __init__(self, code_snippet, language="python", context=None):
        self.code_input = CodeInput(code_snippet=code_snippet, language=language, context=context)
        self.agents = CustomAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Define agents
        code_parser_agent = self.agents.code_parser_agent()
        documentation_writer_agent = self.agents.documentation_writer_agent()
        examples_generator_agent = self.agents.examples_generator_agent()
        final_assembler_agent = self.agents.final_assembler_agent()

        # Define tasks
        code_parsing_task = self.tasks.code_parsing_task(code_parser_agent, self.code_input)
        documentation_writing_task = self.tasks.documentation_writing_task(documentation_writer_agent, self.code_input)
        examples_generation_task = self.tasks.examples_generation_task(examples_generator_agent, self.code_input)
        documentation_assembly_task = self.tasks.documentation_assembly_task(final_assembler_agent, self.code_input)

        # Create the crew
        crew = Crew(
            agents=[
                code_parser_agent,
                documentation_writer_agent,
                examples_generator_agent,
                final_assembler_agent,
            ],
            tasks=[
                code_parsing_task,
                documentation_writing_task,
                examples_generation_task,
                documentation_assembly_task,
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result
    
def run_documentation_generator_crew(args: CodeInput):
    crew = DocumentationGeneratorCrew(args.code_snippet, args.language, args.context)
    results = crew.run()
    return results