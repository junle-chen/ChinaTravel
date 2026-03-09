import os
import time
from openai import OpenAI
from dotenv import load_dotenv

project_root_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_root_path, ".env"))

base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

print(f"Testing API at: {base_url}")
client = OpenAI(base_url=base_url, api_key=api_key, timeout=15)

try:
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-5.1-chat",
        messages=[{"role": "user", "content": "你好，请回复'Hello'"}],
        max_tokens=1000
    )
    print(f"Success in {time.time() - start_time:.2f} seconds!")
    print(response)
    print(f"Type: {type(response)}")
    print(f"Response repr: {repr(response)}")
    
    if hasattr(response, 'choices'):
        print(f"Response content: {response.choices[0].message.content}")
    else:
        print("Response object does not have 'choices' attribute. It might have been returned as a raw string or dictionary.")
except Exception as e:
    print(f"Failed to connect or generate: {e}")
