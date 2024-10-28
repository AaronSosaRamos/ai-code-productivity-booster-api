from app.api.features.doc_generator_assistant.crew import run_documentation_generator_crew
from app.api.features.llm_app_development_assistant.crew import run_llm_development_assistant_crew
from app.api.features.multi_agent_debugging_assistant.crew import run_multi_agent_debugging_crew
from app.api.features.refactoring_assistant.crew import run_refactoring_assistant_crew
from app.api.schemas.llm_app_development_assistant_schema import ApplicationIdea
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
async def refactoring_assistance( data: CodeInput, _ = Depends(key_check)):
    logger.info("Generating the refactoring assistance")
    results = run_refactoring_assistant_crew(data)
    logger.info("The refactoring assistance has been successfully generated")

    return results

@router.post("/doc-generator-assistant")
async def doc_generator_assistance( data: CodeInput, _ = Depends(key_check)):
    logger.info("Generating the documentation generator assistance")
    results = run_documentation_generator_crew(data)
    logger.info("The documentation generator assistance has been successfully generated")

    return results

@router.post("/multi-agent-debugging-assistant")
async def multi_agent_debugging_assistance( data: CodeInput, _ = Depends(key_check)):
    logger.info("Generating the multi-agent debugging assistance")
    results = run_multi_agent_debugging_crew(data)
    logger.info("The documentation generator assistance has been successfully generated")

    return results

@router.post("/llm-app-development-assistant")
async def llm_app_development_assistance( data: ApplicationIdea, _ = Depends(key_check)):
    logger.info("Generating the llm app. development assistance")
    results = run_llm_development_assistant_crew(data)
    logger.info("The llm app. development assistance has been successfully generated")

    return results