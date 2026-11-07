
import ollama

response = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain resume screening in one sentence."
        }
    ]
)

print(response["message"]["content"])
