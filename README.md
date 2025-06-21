#  Smart Email Assistant

A modular, agent-based system that classifies workplace emails, generates context-aware responses using LLMs, and escalates uncertain messages for manual review.

This project leverages:
- A supervised ML model (Logistic Regression) for email classification
- NVIDIA LLaMA (via OpenAI API interface) for natural language response generation
- CrewAI agent architecture for workflow orchestration
- Streamlit app for interactive testing

---

## Features

###  Email Classification Agent (ML-based)
- Classifies input email into: `HR`, `IT`, or `Other`
- Outputs confidence score from the classifier

###  Response Generator Agent (LLM-based)
- Generates a helpful, concise workplace response using an LLM (NVIDIA LLaMA)
- Only triggers when classification confidence ≥ `0.6` and category ≠ `Other`

###  Escalation Agent
- Logs low-confidence or "Other" category emails for manual review
- Escalated emails are saved to `escalation_log.txt`

###  Streamlit App
- Paste and test emails in real time
- Displays classification result, confidence, and generated response
- Flags and logs escalations

---

##  Workflow Logic

```text
1. Input email → ML Classifier
2. If confidence ≥ 0.6 AND category ≠ "Other":
       → LLM generates a response
   Else:
       → Email escalated and logged
```

---

## Project Structure

```text
smart-email-assistant/
│
├── data/
│ └── email_data.csv
│
├── notebooks/
│ └── email_classifier.ipynb
│
├── models/
│ └── email_vectorizer.pkl 
│ └── email_classifier_model.pkl
│
├── agents/
│ ├── email_classifier.py # Classification Agent 
│ ├── response_generator.py # LLM Response Agent
│ └── escalation_agent.py # Escalation Agent
│
├── orchestrator.py # Runs the pipeline using CrewAI
│
├── app.py #  Streamlit app
├── requirements.txt
└── README.md 
```
---
⚙️ Setup Instructions
---
```text
1. Clone the repo
git clone https://github.com/your-username/smart-email-assistant.git
cd smart-email-assistant

2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set NVIDIA API key (replace with yours)
Create a .env file in the root of your project and add your NVIDIA API key:
NVIDIA_API_KEY=your-nvapi-key
```
