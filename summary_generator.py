
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
import json


def generate_summary_chain(api_key: str | None = None):
    """
    Creates a LangChain for generating comprehensive summaries of candidates 
    based on their resume alignment with job requirements
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    print(key)
    if not key:
        raise Exception("OPENAI_API_KEY not found. Provide it via env or pass api_key to generate_summary_chain().")
    # Comprehensive prompt template
    prompt_template = """
You are an expert talent acquisition specialist and HR professional with over 15 years of experience in candidate evaluation across ALL industries and roles. You have a proven track record of identifying top talent and assessing candidate-role fit for positions ranging from entry-level to executive roles across technical, non-technical, creative, sales, marketing, operations, finance, healthcare, and all other domains.

Your task is to analyze a candidate's resume against a specific job description and provide a comprehensive, objective, and insightful summary that helps hiring managers make informed decisions.
Check that the job description is relevant to the resume provided and also job description is properly mentioened not like anything is put there if that's the case simply say
I need a job description to generate a summary and see for the job experience required if the candiate has around that years of experience then only consider him as good fit.
**INPUT INFORMATION:**
Job Description: {job_description}

Candidate Resume: {resume}

**ANALYSIS FRAMEWORK:**
Conduct a thorough analysis using the following methodology:

1. **Requirements Mapping**: Map each job requirement to candidate's experience
2. **Gap Analysis**: Identify missing skills, experience, or qualifications
3. **Strength Assessment**: Highlight areas where candidate exceeds expectations
4. **Cultural Fit Indicators**: Assess alignment with role and company context
5. **Growth Potential**: Evaluate learning ability and career trajectory
6. **Risk Assessment**: Identify potential concerns or red flags

**OUTPUT FORMAT:**

## Candidate Summary Report

### Executive Summary
[2-3 sentence overview of the candidate's overall fit for the role - Strong Fit/Good Fit/Moderate Fit/Poor Fit]

### Candidate Profile
**Name**: [If available, otherwise "Candidate"]
**Current Role**: [Current position and company]
**Total Experience**: [Years of relevant experience]
**Industry Background**: [Primary industry experience]
**Education**: [Relevant educational background]

### Requirements Analysis

#### ✅ **Strong Matches** (Requirements Well Met)
[List job requirements where candidate strongly aligns - provide specific evidence from resume]
- **[Requirement]**: [Specific evidence from resume demonstrating this skill/experience]
- **[Requirement]**: [Specific evidence from resume demonstrating this skill/experience]

#### ⚠️ **Partial Matches** (Requirements Partially Met)
[List job requirements where candidate has some relevant experience but may need development]
- **[Requirement]**: [Evidence from resume + what's missing or needs development]
- **[Requirement]**: [Evidence from resume + what's missing or needs development]

#### ❌ **Gap Areas** (Requirements Not Clearly Met)
[List job requirements where candidate shows little to no evidence]
- **[Requirement]**: [Explain the gap and potential impact]
- **[Requirement]**: [Explain the gap and potential impact]

### Key Strengths
1. **[Strength Category]**: [Specific examples from resume]
2. **[Strength Category]**: [Specific examples from resume]
3. **[Strength Category]**: [Specific examples from resume]

### Areas of Concern
1. **[Concern Category]**: [Specific concern and evidence]
2. **[Concern Category]**: [Specific concern and evidence]

### Experience Relevance Analysis
**Directly Relevant Experience**: [X years/percentage]
**Transferable Experience**: [X years/percentage]
**Industry Alignment**: [High/Medium/Low - with explanation]
**Role Level Alignment**: [Above level/At level/Below level - with explanation]

### Quantifiable Achievements
[List measurable accomplishments from resume with impact assessment]
- **[Achievement]**: [Impact and relevance to role]
- **[Achievement]**: [Impact and relevance to role]


### Overall Recommendation

#### **Fit Score**: [X/10] 
**Reasoning**: [Detailed explanation of the score]

#### **Hiring Recommendation**: 
- [ ] **Strong Yes** - Excellent fit, recommend immediate next steps
- [ ] **Yes** - Good fit, proceed with interview process  
- [ ] **Maybe** - Mixed signals, needs thorough interview assessment
- [ ] **No** - Poor fit for current role, but may fit other positions
- [ ] **Strong No** - Not suitable for organization at this time

**IMPORTANT GUIDELINES:**
- Base all assessments strictly on information provided in the resume
- Be objective and evidence-based in your analysis
- Highlight both strengths and concerns honestly
- Consider the full spectrum of job requirements, not just technical skills
- Account for transferable skills and growth potential
- Provide actionable insights for hiring managers
- Avoid assumptions about gender, age, or other protected characteristics
- Focus on professional qualifications and job-relevant factors only
- If resume lacks detail in certain areas, note this as "insufficient information to assess"
- Tailor analysis complexity to the seniority level of the position
"""

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["job_description", "resume"],
        template=prompt_template
    )
    
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o-mini",
        openai_api_key=key
    )
    
    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )
    
    return chain
