from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CodeInput(BaseModel):
    code_snippet: str
    language: str = Field(default="python", description="Programming language of the code snippet")
    context: Optional[str] = Field(default=None, description="Additional context or comments about the code")

class FunctionElement(BaseModel):
    name: str
    signature: str
    docstring: Optional[str]

class ClassElement(BaseModel):
    name: str
    signature: str
    docstring: Optional[str]
    methods: List[FunctionElement]

class ParsingOutput(BaseModel):
    functions: List[FunctionElement]
    classes: List[ClassElement]
    modules: Optional[List[str]]

class FunctionDocumentation(BaseModel):
    function_name: str
    description: str
    parameters: List[Dict[str, str]]
    return_type: Optional[str]
    examples: Optional[List[str]]

class ClassDocumentation(BaseModel):
    class_name: str
    description: str
    methods: List[FunctionDocumentation]
    attributes: Optional[List[Dict[str, str]]]
    examples: Optional[List[str]]

class ModuleDocumentation(BaseModel):
    module_name: str
    description: str
    functions: List[FunctionDocumentation]
    classes: List[ClassDocumentation]
    examples: Optional[List[str]]

class DocumentationOutput(BaseModel):
    documentation: str