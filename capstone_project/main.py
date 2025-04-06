#Importing Essential Packages for Generative AI
from google import genai
from google.genai import types

#Importing Essential Packages for Website
from IPython.display import HTML, Markdown, display

#genai.__version__
print("Google GenAI version:", genai.__version__)




#Importing Packages for Generative AI Retry Support
from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)



#Linking the Google API Key to the Environment Variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = "gemini-2.0-flash"



from pydantic import BaseModel
#BaseModel helps create the basic JSON Structure layout that we want the GenAI model to respond with.

import gradio as gr
import typing_extensions as typing
from typing import List, Dict, Any

from pydantic import BaseModel, Field
#BaseModel helps create the basic JSON Structure layout that we want the GenAI model to respond with.

import json
import os
from dotenv import load_dotenv






'''
# Define the expected JSON structure
class StudyData(BaseModel):
    title: str
    summary: str
    subtopics: list[str]
'''



from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


#State representing the learner's interaction with the study buddy
class StudyBuddyState(TypedDict):
    # Entire Conversation
    messages: Annotated[list, add_messages]

    # The topics or subjects the learner is currently focusing on.
    focus_topics: list[str]

    # Flag indicating that the study session is considered complete.
    finished: bool


# Assigning Role to the AI Agent
STUDYBUDDY_SYSINT =  (
    "system",
    "You are a helpful and encouraging Virtual Teacher. A student will share a topic they want to learn and you respond back with a very short summary (under 20 words), and provide a numbered list of 3-5 key subtopics that someone learning about the main topic should explore. Also, provide a short title for this learning activity (maximum 5 words).\n\n"
    "Your goal is to support their learning process by making all explanations very simplified and easy to understand. When the student asks a follow-up question, use the context of the previous conversation to provide a relevant and helpful answer.\n\n"
    "You can help the learner by:\n"
    "- Answering their questions about the topics they are studying.\n"
    "- Providing clear and concise explanations of concepts.\n"
    "- Suggesting relevant learning resources like websites, articles, or videos (mentioning that you "
    "  don't have direct access but can suggest where to look).\n"
    "- Helping them break down complex topics into smaller, more manageable parts (the initial subtopics).\n"
    "- Asking clarifying questions to ensure you understand their needs.\n"
    "- Offering encouragement and positive feedback.\n"
    "\n\n"
    "When the student initially provides a topic, your response should be structured to include a title, a brief summary, and a numbered list of subtopics. You will provide this information based on your understanding of the topic. The student may then ask questions about the main topic or any of the subtopics.\n\n"
    "If you are unsure about something, admit that you don't know and suggest ways the learner might find the information (e.g., 'That's a great question! Let's try searching for it online' or "
    "'Perhaps checking a textbook on that subject would be helpful.').\n\n"
    "Once the learner indicates they are finished studying or want to end the session, acknowledge their decision, offer encouragement for their continued learning, and say goodbye!\n\n"
    "Features for managing study topics and generating practice questions are planned for future development.",
)

WELCOME_MSG = "Hi there! I'm your Virtual Study Buddy. What would you like to learn about today?"




#Linking the Google Model to LangGraph
llm = ChatGoogleGenerativeAI(model=genai_model)

# Defining the study_buddy_bot
def chatbot(state: StudyBuddyState) -> StudyBuddyState:
    message_history = [STUDYBUDDY_SYSINT] + state['messages']
    return {"messages": [llm.invoke(message_history)]}


# Initialize the LangGraph
graph_builder = StateGraph(StudyBuddyState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
#graph_builder.add_edge("chatbot", END)


chat_graph = graph_builder.compile()

conversation_history = []

#print("chatbot initialized")




def study_buddy(input_text):

    state = chat_graph.invoke({"messages": [input_text]})

    # Access the AIMessage (assuming it's the last one)
    ai_response = state['messages'][-1].content
    return ai_response





#Gradio Interface
if 'iface' in locals():
    iface.close()

iface = gr.Interface(
    fn=study_buddy,
    inputs=gr.Textbox(label="What would you like to learn about today?"),
    outputs="text",
    title="Your Study Buddy (LangGraph)",
    description="Enter a topic you'd like to learn about."
)

iface.launch(share=False)