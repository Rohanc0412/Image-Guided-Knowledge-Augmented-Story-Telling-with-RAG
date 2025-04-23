import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI( api_key= OPENAI_API_KEY)

def query_model(caption, summary):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a storytelling AI. Create a highly detailed, elaborate, and comprehensive "
                "fictional but fact-based story inspired by the given image caption and event summary. "
                "If the factual background is 'No article found', you are free to draw on your own creative "
                "insights and historical context. Compose a compelling story in three acts that includes extensive "
                "character development, rich dialogue, vivid descriptions, and in-depth narrative details. "
                "The story should be no less than 3000 characters in length."
            )
        },
        {
            "role": "user",
            "content": (
                f"Image Caption: \"{caption}\"\n\n"
                f"Factual Background: \"{summary}\"\n\n"
                "Story:"
            )
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_completion_tokens=3000,
            temperature=0.7,
        )
        
        generated_text = response.choices[0].message.content
    except Exception as e:
        print("Error during story generation:", e)
        generated_text = "Story generation failed."
    
    return generated_text
