from app.api.features.doc_generator_assistant.crew import run_documentation_generator_crew
from app.api.features.refactoring_assistant.crew import run_refactoring_assistant_crew
from app.api.schemas.refactoring_assistant_schema import CodeInput
from fastapi import APIRouter, Depends
from app.api.logger import setup_logger
from app.api.auth.auth import key_check

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/refactoring-assistant")
async def submit_tool( data: CodeInput, _ = Depends(key_check)):
    logger.info("Generating the refactoring assistance")
    results = run_refactoring_assistant_crew(data)
    logger.info("The refactoring assistance has been successfully generated")

    return results

@router.post("/doc-generator-assistant")
async def submit_tool( data: CodeInput, _ = Depends(key_check)):
    logger.info("Generating the documentation generator assistance")
    results = run_documentation_generator_crew(data)
    logger.info("The documentation generator assistance has been successfully generated")

    return results