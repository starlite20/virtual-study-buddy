import os
import json

from dotenv import load_dotenv

from pydantic import BaseModel
#BaseModel helps create the basic JSON Structure layout that we want the GenAI model to respond with.

import gradio as gr
import typing_extensions as typing

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



# Define the expected JSON structure
class StudyData(BaseModel):
    title: str
    summary: str
    subtopics: list[str]

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
        prompt = f"""Summarize the key concepts of '{topic}' in less than 20 words. Then, suggest 3-5 key subtopics for someone learning about '{topic}' should explore. Additionally have a short title for this learning activity with a maximum of 5 words. Return your response as a JSON object with the following structure:
                    {{
                      "title": "...",
                      "summary": "...",
                      "subtopics": ["subtopic 1", "subtopic 2", ...]
                    }}"""
        generation_config = {
            'response_mime_type': 'application/json',
            'response_schema': StudyData,
        }

        try:
            gemini_response = client.models.generate_content(
                model=genai_model,
                contents=[prompt],
                config=generation_config
            )
            
            # Access the parsed data directly
            study_data: StudyData = gemini_response.parsed
            study_buddy.current_topic = study_data.title
            search_summary = study_data.summary
            study_buddy.subtopics_list = study_data.subtopics

            subtopics_display = "\n".join([f"{i+1}. {sub}" for i, sub in enumerate(study_buddy.subtopics_list)]) if study_buddy.subtopics_list else "No subtopics suggested yet."

            response = f"{study_buddy.current_topic}:\n\n{search_summary}\n\nHere are some subtopics we can explore:\n{subtopics_display}\n\nType a subtopic to learn more, or say 'Tell me more' to start with the first one."

        except json.JSONDecodeError as e:
            response = f"Error decoding JSON response from Gemini: {e}\n\nRaw response: {gemini_response.text}"
            study_buddy.subtopics_list = []
        except Exception as e:
            response = f"Error during Gemini interaction: {e}"
            study_buddy.subtopics_list = []

        
    else:
        user_input = input_text.lower()
        if user_input == "tell me more":
            if study_buddy.subtopics_list:
                first_subtopic = study_buddy.subtopics_list[0]
                subtopic_prompt = f"Explain '{first_subtopic}' in simple terms."
                try:
                    subtopic_response = client.models.generate_content(
                        model=genai_model,
                        contents=subtopic_prompt
                    )
                    response = f"Okay, let's start with '{first_subtopic}':\n\n{subtopic_response.text}"
                except Exception as e:
                    response = f"Error explaining subtopic: {e}"
            else:
                response = "Sorry, no subtopics available to explore further."
        elif user_input in [sub.lower() for sub in study_buddy.subtopics_list]:
            selected_subtopic = study_buddy.subtopics_list[[sub.lower() for sub in study_buddy.subtopics_list].index(user_input)]
            subtopic_prompt = f"Explain '{selected_subtopic}' in simple terms."
            try:
                subtopic_response = client.models.generate_content(
                    model=genai_model,
                    contents=subtopic_prompt
                )
                response = f"Let's dive into '{selected_subtopic}':\n\n{subtopic_response.text}"
            except Exception as e:
                response = f"Error explaining subtopic: {e}"
        else:
            response = f"Sorry, I didn't understand. You can type a subtopic from the list or say 'Tell me more'."

    return response

if 'iface' in locals():
    iface.close()

iface = gr.Interface(
    fn=study_buddy,
    inputs=gr.Textbox(label="What would you like to learn today?"),
    outputs="text",
    title="Your Study Buddy",
    description="Enter a topic you'd like to learn about."
)

iface.launch(share=False)