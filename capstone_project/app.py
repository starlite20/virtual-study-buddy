from dotenv import load_dotenv
import os

import gradio as gr

#Importing Essential Packages for Generative AI
from google import genai
from google.genai import types

#Importing Essential Packages for Website
from IPython.display import HTML, Markdown, display

#genai.__version__
print("Google GenAI version:", genai.__version__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = "gemini-2.0-flash"



def study_buddy(input_text):
    global current_topic
    if not hasattr(study_buddy, 'current_topic'):
        study_buddy.current_topic = None
        study_buddy.subtopics_list = []
        
    response = ""

    if not study_buddy.current_topic:
        study_buddy.current_topic = input_text
        topic = study_buddy.current_topic

        # Simulate web search and summarization with Gemini
        prompt = f"Summarize the key concepts of '{topic}' in a few sentences as if you found relevant information on the web. If possible, briefly mention the types of sources where this information might be found (e.g., educational websites, scientific articles)."
        try:
            gemini_response = client.models.generate_content(
                model=genai_model,
                contents=prompt
            )
            search_summary = gemini_response.text
        except Exception as e:
            search_summary = f"Error during Gemini summarization: {e}"

        introduction = f"Let's learn about {topic}! 🚀"
        summary_points_list = [] # We'll generate these later based on the summary
        study_buddy.subtopics = {} # We'll populate these later

        response = f"{introduction}\n\nHere's a brief overview:\n{search_summary}\n\nTell me more to dive deeper."

    else:
        user_input = input_text
        # For now, we'll just acknowledge the follow-up
        response = f"Okay, you want to know more about {study_buddy.current_topic}."

    return response

if 'iface' in locals():
    iface.close()

iface = gr.Interface(
    fn=study_buddy,
    inputs=gr.Textbox(label="What do you want to learn today?"),
    outputs="text",
    title="Virtual Study Buddy",
    description="Enter a topic you'd like to learn about."
)

iface.launch(share=False)