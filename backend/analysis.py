import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file, useful for local testing
load_dotenv()

# Get Groq API key from Streamlit Cloud Secrets or local environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

# Stop the app if API key is missing
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets.")
    st.stop()

# Set Groq API key for LangChain/Groq
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize Groq model
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)


def analyze_resume(full_resume, job_description):
    template = """
You are an AI assistant specialized in resume analysis and recruitment.

Analyze the given resume and compare it with the job description.

Give the response in this format:

**OVERVIEW**
- **Match Percentage**: Give an estimated match percentage between the resume and job description.
- **Matched Skills**: List the skills from the job description that are found in the resume.
- **Unmatched Skills**: List the skills from the job description that are missing from the resume.

**DETAILED ANALYSIS**
1. Explain the overall match between the resume and the job description.
2. Explain which skills and experiences match well.
3. Explain which skills, tools, or experiences are missing.
4. Suggest how the resume can be improved for this job.

**ADDITIONAL COMMENTS**
Give final advice for the candidate, recruiter, or HR manager.

Resume:
{resume}

Job Description:
{job_description}

Analysis:
"""

    prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template=template
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": full_resume,
        "job_description": job_description
    })

    return response.content
