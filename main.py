from fastapi import FastAPI, UploadFile, Form, HTTPException, Header
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from models import SummaryResponse, QuestionsResponse
import pymupdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from question_generator import generate_questions_chain
from summary_generator import generate_summary_chain

app = FastAPI(
    title="AI Interview Assistant",
    description="Generate candidate summaries and interview questions using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    # Accept either "Bearer <key>" or direct key in header
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return auth_header.strip()

def extract_text_with_pymupdf(file, filename: str) -> str:
    try:
        file.seek(0)
        if filename.lower().endswith('.pdf'):
            pdf_bytes = file.read()
            pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            text_content = [pdf_document[page_num].get_text() for page_num in range(pdf_document.page_count)]
            pdf_document.close()
            return "\n".join(text_content)
        elif filename.lower().endswith('.txt'):
            content = file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return str(content)
        else:
            raise Exception(f"Unsupported file type: {filename}")
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {str(e)}")
        raise Exception(f"Failed to process file: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the AI Interview Assistant API. Use /docs for API documentation."}
@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(
    resume: UploadFile,
    job_description: str = Form(...),
    authorization: str | None = Header(None)
) -> Dict[str, Any]:
    openai_api_key = _extract_bearer_token(authorization)
    if not openai_api_key:
         raise HTTPException(status_code=401, detail="OpenAI API key missing or invalid")
    if not resume or not job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description are required")

    filename = resume.filename.lower() if resume.filename else ""
    if not any(filename.endswith(ext) for ext in ['.pdf', '.txt']):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or TXT only.")

    resume_text = extract_text_with_pymupdf(resume.file, resume.filename)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume.")

    summary_chain = generate_summary_chain(api_key=openai_api_key)
    summary = summary_chain.run({
        "resume": resume_text,
        "job_description": job_description
    })

    return {
        "summary": summary.strip()
    }

@app.post("/generate-questions",response_model=QuestionsResponse)
async def generate_questions(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    interview_type: str = Form(...),
    authorization: str | None = Header(None)
):
     
    openai_api_key = _extract_bearer_token(authorization)
    if not openai_api_key:
         raise HTTPException(status_code=401, detail="OpenAI API key missing or invalid")
    if not resume_text.strip() or not job_description.strip() or not interview_type.strip():
        raise HTTPException(status_code=400, detail="All fields are required")

    question_chain = generate_questions_chain(api_key=openai_api_key)
    questions_raw = question_chain.run({
        "resume": resume_text,
        "job_description": job_description,
        "interview_type": interview_type
    })

    questions = []
    for line in questions_raw.split('\n'):
        line = line.strip()
        if line and len(line) > 10:
            if line[0].isdigit() and '.' in line[:5]:
                line = line.split('.', 1)[1].strip()
            questions.append(line)

    if not questions:
        questions = ["Could you walk me through your relevant experience for this role?"]

    return {
        "questions": questions
    }

@app.get("/health")
async def health_check():
    """Extended health check with API key validation"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "OpenAI API key not configured"}
        
        return {
            "status": "healthy",
            "message": "All systems operational",
            "openai_configured": bool(api_key)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port,reload=True)
