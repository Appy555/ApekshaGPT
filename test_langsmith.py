from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

@traceable(name="My First LangSmith Test")
def say_hello():
    return "Hello from my Agentic Chatbot!"

result = say_hello()
print(result)