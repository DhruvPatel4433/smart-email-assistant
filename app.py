import streamlit as st
import pickle
import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai.llm import LLM
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

llm = LLM(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

with open("email_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("email_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@tool("classify_email")
def classify_email(email_text: str) -> str:
    """Classify an email and return category with confidence score"""
    try:
        vec = vectorizer.transform([email_text])
        probs = model.predict_proba(vec)
        pred = model.predict(vec)[0]
        confidence = round(probs.max(), 2)
        return json.dumps({
            "category": pred,
            "confidence": confidence,
            "email_text": email_text
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "category": "Other",
            "confidence": 0.0
        })

@tool("generate_response")
def generate_email_response(email_text: str, category: str) -> str:
    """Generate a professional email response"""
    try:
        prompt = f"""You are a helpful workplace assistant.

Email: "{email_text}"
Category: {category}

Generate a professional, helpful response to this email. Keep it concise."""
        response = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

@tool("log_escalation")
def log_for_escalation(email_text: str, category: str, confidence: float) -> str:
    """Log email for human escalation"""
    try:
        record = {
            "timestamp": datetime.now().isoformat(),
            "email_text": email_text,
            "predicted_category": category,
            "confidence": confidence,
            "status": "needs_human_review"
        }
        with open("escalation_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return f"Email logged for escalation. Category: {category}, Confidence: {confidence}"
    except Exception as e:
        return f"Error logging escalation: {str(e)}"

email_classifier = Agent(
    role="Email Classifier",
    goal="Classify emails accurately and determine confidence levels",
    backstory="You are an expert at categorizing workplace emails and assessing classification confidence.",
    tools=[classify_email],
    llm=llm,
    verbose=True
)

response_generator = Agent(
    role="Response Writer",
    goal="Create professional email responses",
    backstory="You write helpful, professional email responses for various workplace situations.",
    tools=[generate_email_response],
    llm=llm,
    verbose=True
)

escalation_manager = Agent(
    role="Escalation Handler",
    goal="Handle emails that need human review",
    backstory="You manage emails that require human attention and ensure proper logging.",
    tools=[log_for_escalation],
    llm=llm,
    verbose=True
)

def process_email(email_text: str):
    classification_task = Task(
        description=f"Classify this email: '{email_text}'",
        agent=email_classifier,
        expected_output="JSON with category, confidence, and email_text"
    )
    classification_crew = Crew(
        agents=[email_classifier],
        tasks=[classification_task],
        verbose=False
    )
    classification_crew.kickoff()
    classification_json = classify_email.run(email_text=email_text)
    classification_data = json.loads(classification_json)

    category = classification_data.get("category")
    confidence = classification_data.get("confidence", 0.0)

    if confidence >= 0.6 and category != "Other":
        response_task = Task(
            description=f"Generate a professional response for this {category} email: '{email_text}'",
            agent=response_generator,
            expected_output="Professional email response"
        )
        response_crew = Crew(
            agents=[response_generator],
            tasks=[response_task],
            verbose=False
        )
        response_crew.kickoff()
        generated_response = generate_email_response.run(email_text=email_text, category=category)

        return {
            "status": "auto_responded",
            "predicted_category": category,
            "confidence": confidence,
            "response": generated_response
        }
    else:
        escalation_task = Task(
            description=f"Log this email for escalation: '{email_text}' (Category: {category}, Confidence: {confidence})",
            agent=escalation_manager,
            expected_output="Escalation confirmation"
        )
        escalation_crew = Crew(
            agents=[escalation_manager],
            tasks=[escalation_task],
            verbose=False
        )
        escalation_crew.kickoff()
        log_result = log_for_escalation.run(email_text=email_text, category=category, confidence=confidence)

        return {
            "status": "escalated",
            "predicted_category": category,
            "confidence": confidence,
            "log_result": log_result
        }

st.set_page_config(page_title="Smart Email Assistant", layout="centered")

st.title("Smart Email Assistant")

email_input = st.text_area("Enter an email:", height=200)

if st.button("Process Email"):
    if not email_input.strip():
        st.warning("Please enter an email to process.")
    else:
        result = process_email(email_input)

        st.subheader("Result")
        st.write(f"**Status**: {result['status'].upper()}")
        st.write(f"**Category**: {result['predicted_category']}")
        st.write(f"**Confidence**: {result['confidence']}")

        if result['status'] == 'auto_responded':
            st.subheader("Generated Response")
            st.success(result['response'])
        else:
            st.subheader("Escalation Log")
            st.info(result['log_result'])
