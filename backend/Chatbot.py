


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
        raise ValueError(f"Missing or empty environment variable: {var}")

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


System = f"""Hello, I am {Username}. You are a very accurate and advanced AI chatbot named {Assistantname} with real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

SystemChatBot = [{"role": "system", "content": System}]

def RealtimeInformation():
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

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def ChatBot(Query):
    """Sends the user's query to the chatbot and returns the AI's response"""
    try:
        with open(CHAT_LOG_PATH, "r") as f:
            messages = load(f)
        messages.append({"role": "user", "content": Query})

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                Answer += content

        Answer = Answer.replace("</s>", "")
        messages.append({"role": "assistant", "content": Answer})
        with open(CHAT_LOG_PATH, "w") as f:
            dump(messages, f, indent=4)
        return AnswerModifier(Answer)

    except Exception as e:
        print(f"Error: {e}")
        with open(CHAT_LOG_PATH, "w") as f:
            dump([], f, indent=4)
        return f"Error occurred: {e}. Chat log reset. Please try again."

if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Question: ")
        print(ChatBot(user_input))