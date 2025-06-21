import json
from datetime import datetime
from crewai import Agent
from crewai.tools import tool

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


def get_escalation_agent(llm):
    return Agent(
        role="Escalation Handler",
        goal="Handle emails that need human review",
        backstory="You manage emails that require human attention and ensure proper logging.",
        tools=[log_for_escalation],
        llm=llm,
        verbose=True
    )
