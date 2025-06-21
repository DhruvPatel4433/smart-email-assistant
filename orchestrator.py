import os
import json
import pickle
from datetime import datetime
from dotenv import load_dotenv
from crewai import Task, Crew
from crewai.llm import LLM
from openai import OpenAI

from agents.email_classifier import get_email_classifier
from agents.response_generator import get_response_generator
from agents.escalation_agent import get_escalation_agent

_ = load_dotenv()

with open("models/email_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/email_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

llm = LLM(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

email_classifier = get_email_classifier(llm, model, vectorizer)
response_generator = get_response_generator(llm, client)
escalation_manager = get_escalation_agent(llm)

def process_email(email_text: str):
    print(f"\nProcessing email: '{email_text[:50]}...'")

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

    classification_result = email_classifier.tools[0].run(email_text=email_text)
    classification_data = json.loads(classification_result)

    category = classification_data.get("category")
    confidence = classification_data.get("confidence", 0.0)

    print(f"Classification: {category} (confidence: {confidence})")

    if confidence >= 0.6 and category != "Other":
        print("Auto-responding...")

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

        response = response_generator.tools[0].run(email_text=email_text, category=category)

        return {
            "status": "auto_responded",
            "email_text": email_text,
            "predicted_category": category,
            "confidence": confidence,
            "response": response
        }
    else:
        print("Escalating for human review...")

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

        log_result = escalation_manager.tools[0].run(
            email_text=email_text,
            category=category,
            confidence=confidence
        )

        return {
            "status": "escalated",
            "email_text": email_text,
            "predicted_category": category,
            "confidence": confidence,
            "reason": "Low confidence or unknown category",
            "log_result": log_result
        }

if __name__ == "__main__":
    test_emails = [
        "Hi, I forgot my laptop password. Please help."
    ]

    print("Starting Email Processing System with CrewAI")

    for email in test_emails:
        result = process_email(email)
        print(f"Result: {result['status'].upper()}")
        if result["status"] == "auto_responded":
            print(f"Response:\n{result['response'][:600]}")
        
