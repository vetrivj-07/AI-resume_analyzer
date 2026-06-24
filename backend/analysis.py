import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Get Groq API key from Streamlit Secrets or environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize the ChatGroq model
llm = ChatGroq(model_name="mistral-saba-24b")


def analyze_resume(full_resume, job_description):
    template = """
    You are an AI assistant specialized in resume analysis and recruitment. Analyze the given resume and compare it with the job description. 
    
    Example Response Structure:
    
    **OVERVIEW**:
    - **Match Percentage**: [Calculate overall match percentage between the resume and job description]
    - **Matched Skills**: [List the skills in job description that match the resume]
    - **Unmatched Skills**: [List the skills in the job description that are missing in the resume]

    **DETAILED ANALYSIS**:
    Provide a detailed analysis about:
    1. Overall match percentage between the resume and job description
    2. List of skills from the job description that match the resume
    3. List of skills from the job description that are missing in the resume
    
    **Additional Comments**:
    Additional comments about the resume and suggestions for the recruiter or HR manager.

    Resume: {resume}
    Job Description: {job_description}

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
