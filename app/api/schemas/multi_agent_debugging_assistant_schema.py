from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CodeInput(BaseModel):
    code_snippet: str
    language: str = Field(default="python", description="Programming language of the code snippet")
    context: Optional[str] = Field(default=None, description="Additional context or comments about the code")
    dependencies: Optional[List[str]] = Field(default=None, description="List of dependencies or libraries used")
    environment: Optional[str] = Field(default=None, description="Execution environment details")
    expected_behavior: Optional[str] = Field(default=None, description="Description of the expected behavior of the code")
    actual_behavior: Optional[str] = Field(default=None, description="Description of the actual behavior observed")
    inputs: Optional[List[str]] = Field(default=None, description="Expected inputs to the code")
    outputs: Optional[List[str]] = Field(default=None, description="Expected outputs from the code")

class BugDetail(BaseModel):
    bug_id: int
    description: str
    line_number: Optional[int]
    severity: str
    error_message: Optional[str]
    stack_trace: Optional[str]
    variables_at_fault: Optional[List[str]]
    conditions: Optional[str]
    frequency: Optional[str]
    module: Optional[str]
    function: Optional[str]
    code_snippet: Optional[str]
    replication_steps: Optional[List[str]]
    logs: Optional[List[str]]
    environment: Optional[str]
    timestamp: Optional[str]

class AnalysisOutput(BaseModel):
    bugs: List[BugDetail]
    warnings: Optional[List[str]]
    error_counts: Optional[Dict[str, int]]
    execution_time: Optional[float]
    resource_usage: Optional[Dict[str, float]]
    code_complexity: Optional[float]
    code_coverage: Optional[float]
    test_results: Optional[Dict[str, str]]
    static_analysis_reports: Optional[Dict[str, str]]
    dependency_issues: Optional[List[str]]

class DebuggingStep(BaseModel):
    step_id: int
    description: str
    related_bug_ids: List[int]
    priority: str
    estimated_time_hours: Optional[float]
    required_resources: Optional[List[str]]
    dependencies: Optional[List[int]]
    assigned_to: Optional[str]
    tools_needed: Optional[List[str]]
    risk_assessment: Optional[str]
    success_criteria: Optional[str]

class DebuggingPlan(BaseModel):
    steps: List[DebuggingStep]
    overall_priority: str
    total_estimated_time_hours: Optional[float]
    risks: Optional[List[str]]
    assumptions: Optional[List[str]]
    contingencies: Optional[List[str]]

class FixSuggestion(BaseModel):
    suggestion_id: int
    bug_id: int
    suggestion: str
    detailed_steps: List[str]
    code_changes: Optional[Dict[str, str]]  # e.g., { "line 42": "change x to y" }
    potential_side_effects: Optional[List[str]]
    testing_required: Optional[List[str]]
    estimated_effort_hours: Optional[float]
    dependencies_affected: Optional[List[str]]
    alternative_solutions: Optional[List[str]]
    references: Optional[List[str]]

class FixSuggestions(BaseModel):
    suggestions: List[FixSuggestion]

class FixedCode(BaseModel):
    code_snippet: str
    changes_made: Dict[str, str]
    bugs_fixed: List[int]
    new_dependencies: Optional[List[str]]
    tests_performed: Optional[List[str]]
    performance_improvements: Optional[Dict[str, float]]
    remaining_issues: Optional[List[str]]
    code_quality_metrics: Optional[Dict[str, float]]
    documentation_updates: Optional[List[str]]