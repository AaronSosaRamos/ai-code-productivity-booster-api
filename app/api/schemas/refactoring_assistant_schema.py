from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CodeInput(BaseModel):
    code_snippet: str
    language: str = Field(default="python", description="Programming language of the code snippet")
    context: Optional[str] = Field(default=None, description="Additional context or comments about the code")

class IssueDetail(BaseModel):
    issue_id: int
    description: str
    line_number: Optional[int]
    severity: str
    suggestion: Optional[str]

class AnalysisOutput(BaseModel):
    issues: List[IssueDetail]
    complexity_score: float
    cyclomatic_complexity: Optional[int]
    maintainability_index: Optional[float]
    technical_debt: Optional[float]

class RefactoringOpportunity(BaseModel):
    opportunity_id: int
    description: str
    related_issues: List[int]
    priority: str

class RefactoringOpportunities(BaseModel):
    opportunities: List[RefactoringOpportunity]

class SuggestionDetail(BaseModel):
    opportunity_id: int
    suggestion: str
    estimated_effort_hours: Optional[float]
    dependencies_affected: Optional[List[str]]

class RefactoringSuggestions(BaseModel):
    suggestions: List[SuggestionDetail]

class RefactoredCode(BaseModel):
    code_snippet: str
    changes_made: Dict[str, str]
    new_dependencies: Optional[List[str]]