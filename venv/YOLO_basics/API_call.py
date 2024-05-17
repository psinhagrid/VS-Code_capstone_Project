import os
from groq import Groq


# Instantiate the Groq client object with the API key
client = Groq(api_key='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')

completion = client.chat.completions.create(
    model="gemma-7b-it",
    messages=[
        {
            "role": "user",
            "content": "Which is the scariest creature in subnautica other than guardian titan ? \n"
        },
        {
            "role": "assistant",
            "content": "**Guardian Titan** is the biggest creature in Subnautica."
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
