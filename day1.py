

from groq import Groq

client = Groq(api_key="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")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "recursion explain",
        }
    ],
    model="llama-3.3-70b-versatile",
)


print(chat_completion.choices[0].message.content)


