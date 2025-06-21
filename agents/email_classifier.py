import json
from crewai import Agent
from crewai.tools import tool


def create_classify_tool(model, vectorizer):
    @tool("classify_email")
    def classify_email(email_text: str) -> str:
        """Classify an email and return category with confidence score"""
        try:
            vec = vectorizer.transform([email_text])
            probs = model.predict_proba(vec)
            pred = model.predict(vec)[0]
            confidence = round(probs.max(), 2)

            result = {
                "category": pred,
                "confidence": confidence,
                "email_text": email_text
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e), "category": "Other", "confidence": 0.0})
    return classify_email

def get_email_classifier(llm, model, vectorizer):
    classify_tool = create_classify_tool(model, vectorizer)
    return Agent(
        role="Email Classifier",
        goal="Classify emails accurately and determine confidence levels",
        backstory="You are an expert at categorizing workplace emails and assessing classification confidence.",
        tools=[classify_tool],
        llm=llm,
        verbose=True
    )
