from crewai import Agent
from crewai.tools import tool

def create_response_tool(client):
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
    return generate_email_response


def get_response_generator(llm, client):
    tool = create_response_tool(client)
    return Agent(
        role="Response Writer",
        goal="Create professional email responses",
        backstory="You write helpful, professional email responses for various workplace situations.",
        tools=[tool],
        llm=llm,
        verbose=True
    )
