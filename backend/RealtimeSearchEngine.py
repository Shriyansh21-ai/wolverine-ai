from googlesearch import search
from groq import Groq
from json import load, dump, JSONDecodeError
import datetime
import os
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
required_vars = ["Username", "Assistantname", "GroqAPIKey"]
for var in required_vars:
    if not env_vars.get(var):
        raise ValueError(f"Missing or empty environment variable: {var} in .env file")

Username = env_vars["Username"]
Assistantname = env_vars["Assistantname"]
GroqAPIKey = env_vars["GroqAPIKey"]

try:
    client = Groq(api_key=GroqAPIKey)
except Exception as e:
    raise ValueError(f"Failed to initialize Groq client: {e}")

CHAT_LOG_PATH = os.path.join("Data", "ChatLog.json")

try:
    with open(CHAT_LOG_PATH, "r") as f:
        messages = load(f)
except (FileNotFoundError, JSONDecodeError):
    os.makedirs(os.path.dirname(CHAT_LOG_PATH), exist_ok=True)
    with open(CHAT_LOG_PATH, "w") as f:
        dump([], f)
    messages = []

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""

def GoogleSearch(query):
    try:
        results = list(search(query, advanced=True, num_results=5))
        Answer = f"The search results for '{query}' are:\n[start]\n"
        for i in results:
            Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"
        Answer += "[end]"
        return Answer
    except Exception as e:
        return f"Google search failed: {e}"

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can I help you?"}
]

def Information():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    data = f"Please use this real-time information if needed:\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second} seconds"
    return data

def RealtimeSearchEngine(prompt):
    global SystemChatBot, messages
    try:
        with open(CHAT_LOG_PATH, "r") as f:
            messages = load(f)
        
    except Exception as e:
        print(f"Failed to load chat log: {e}")
        messages = []

    messages.append({"role": "user", "content": f"{prompt}"})
    search_result = GoogleSearch(prompt)
    SystemChatBot.append({"role": "system", "content": search_result})
    

    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role": "system", "content": Information()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )
    except Exception as e:
        print(f"Groq API error: {e}")
        SystemChatBot.pop()
        return "Failed to get response from Groq API."

    Answer = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            Answer += content
    
    Answer = Answer.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})
    try:
        with open(CHAT_LOG_PATH, "w") as f:
            dump(messages, f, indent=4)
    except Exception as e:
        print(f"Failed to save chat log: {e}")
    
    SystemChatBot.pop()
    
    return AnswerModifier(Answer)

if __name__ == "__main__":
    while True:
        prompt = input("Enter Your query (type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        print(RealtimeSearchEngine(prompt))