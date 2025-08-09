from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

def generate_questions_chain(api_key: str | None = None):
    """
    Creates a LangChain for generating tailored interview questions for ANY role/industry
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
     raise Exception("OPENAI_API_KEY not found. Provide it via env or pass api_key to generate_questions_chain().")
    # Comprehensive prompt template
    prompt_template = """
You are an expert HR professional and seasoned interviewer with extensive experience in conducting interviews across ALL industries, roles, and seniority levels - from entry-level to C-suite positions, across technical, non-technical, creative, sales, marketing, operations, finance, healthcare, education, and all other domains. Your task is to generate highly relevant, insightful interview questions based on the provided information.

**INPUT INFORMATION:**
Job Description: {job_description}

Candidate Resume: {resume}

Interview Type: {interview_type}

**INSTRUCTIONS:**
Analyze the job description and candidate's resume and the interview type carefully, then generate interview questions that are:
1. **Relevant**: Directly aligned with the job requirements and candidate's background
2. **Insightful**: Designed to reveal the candidate's true capabilities, experience, and fit
3. **Differentiated**: Tailored specifically to the interview type specified
4. **Progressive**: Ranging from basic to advanced difficulty levels
5. **Actionable**: Questions that allow for specific, measurable responses
6. **Generate questions that are only meant for {interview_type} interviews. Do not generate questions for other interview types.

**INTERVIEW TYPE GUIDELINES:**

**If TECHNICAL Interview:**
- Focus on role-specific technical skills, tools, and methodologies
- Include practical scenarios, problem-solving exercises, or technical challenges relevant to the field
- Assess depth of knowledge in technologies, processes, or methods mentioned in job description
- Include questions about best practices, optimization, and troubleshooting within the domain
- Ask about specific projects, implementations, or achievements from their resume
- Adapt technical complexity based on the role (e.g., software engineering vs. technical writing vs. data analysis)

**If HR/BEHAVIORAL Interview:**
- Focus on soft skills, cultural fit, and behavioral competencies
- Use STAR method framework (Situation, Task, Action, Result) questions
- Assess communication, teamwork, adaptability, and work ethic
- Explore career motivation, goals, and company alignment
- Include scenario-based questions about workplace challenges relevant to any industry
- Focus on interpersonal skills and emotional intelligence

**If MANAGERIAL/LEADERSHIP Interview:**
- Focus on leadership experience, team management, and strategic thinking
- Assess decision-making, delegation, and performance management skills
- Include questions about budget management, project delivery, and stakeholder management
- Explore change management and team development experience
- Adapt to different management contexts (people management, project management, etc.)

**If CULTURAL FIT Interview:**
- Focus on values alignment, work style preferences, and team dynamics
- Assess adaptability, company culture understanding, and long-term commitment
- Include questions about work-life balance, collaboration style, and feedback reception
- Explore alignment with company mission and values regardless of industry

**If CASE STUDY/PROBLEM-SOLVING Interview:**
- Present business scenarios or case studies relevant to the specific role and industry
- Assess analytical thinking, problem-solving methodology, and communication
- Focus on structured thinking and practical application of knowledge
- Adapt scenarios to the role context (sales scenarios for sales roles, customer service scenarios for CS roles, etc.)

**If INDUSTRY-SPECIFIC Interview:**
- Focus on industry knowledge, regulations, trends, and best practices
- Include questions about industry challenges and opportunities
- Assess understanding of market dynamics and competitive landscape
- Explore experience with industry-specific tools, processes, or methodologies

**OUTPUT FORMAT:**
Provide your response in the following structured format:
Give top 5-10 questions for the interview type specified, only questions, no explanations or additional text.

**Additional Recommendations:**
[Any specific advice for conducting this interview based on the candidate's profile]

GIVE ONLY the questions in the output, without any additional commentary or explanations.

Remember to:
- Reference specific experiences, skills, or achievements from the candidate's resume
- Align questions with the specific requirements mentioned in the job description
- Ensure questions are appropriate for the seniority level and industry of the position
- Adapt technical depth and complexity based on the role type (technical vs. non-technical)
- Consider the company culture and role expectations
- Generate questions suitable for ANY industry or role type
- Focus on role-relevant skills rather than assuming technical background
- Tailor the complexity and terminology to match the position level and field
"""
    prompt = PromptTemplate(
        input_variables=["job_description", "resume", "interview_type"],
        template=prompt_template
    )
    
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0.2, 
        model_name="gpt-4o-mini",
        openai_api_key=key
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )
    
    return chain
