#Importing Essential Packages for Generative AI
from google import genai
from google.genai import types

#Importing Packages for Generative AI Retry Support
from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)




import os
from dotenv import load_dotenv


#Linking the Google API Key to the Environment Variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = "gemini-2.0-flash"





import gradio as gr
import typing_extensions as typing
from typing import List, Dict, Any

from pydantic import BaseModel, Field
#BaseModel helps create the basic JSON Structure layout that we want the GenAI model to respond with.

import json





from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage

from typing import Literal

from pprint import pprint






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





#AI Model Initialization for LangGraph
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


#initiating state of chatbot
def chatbot_welcome(state: StudyBuddyState) -> StudyBuddyState:
    if state["messages"]:
        # If there are messages, continue the conversation with the Gemini model.
        new_output = llm.invoke([STUDYBUDDY_SYSINT] + state['messages'])
    else:
        # If there are no messages, start with the welcome message.
        new_output = AIMessage(content=WELCOME_MSG)

    return state | {"messages": [new_output]}


# Defining the study_buddy_bot
def chatbot(state: StudyBuddyState) -> StudyBuddyState:
    message_history = [STUDYBUDDY_SYSINT] + state['messages']
    return {"messages": [llm.invoke(message_history)]}


#user input
def human_node(state: StudyBuddyState) -> StudyBuddyState:
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    # Get the last user message (already included in the state)
    user_message = state["messages"][-1]

    # Extract actual string if the message is a tuple or HumanMessage
    user_input = ""
    if isinstance(user_message, tuple):
        role, user_input = user_message
    elif hasattr(user_message, "content"):
        user_input = user_message.content

    if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state


def maybe_exit_human_node(state: StudyBuddyState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"




# Initialize the LangGraph
graph_builder = StateGraph(StudyBuddyState)


# Add the chatbot and human nodes to the app graph.
graph_builder.add_node("chatbot", chatbot_welcome)
graph_builder.add_node("human", human_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human");
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

#graph_builder.add_edge("chatbot", END)


chat_graph = graph_builder.compile()

conversation_history = []

print("okay")








config = {"recursion_limit": 100}


def study_buddy(message: str, history: list) -> tuple[str, list]:
    global chat_graph

    # Prepare LangGraph input with the current message and history
    chat_history_langchain_format = []
    for user_msg, bot_response in history:
        chat_history_langchain_format.append(HumanMessage(content=user_msg))
        if bot_response:
            chat_history_langchain_format.append(AIMessage(content=bot_response))

    # Add the new message from this turn
    chat_history_langchain_format.append(HumanMessage(content=message))

    # Invoke LangGraph with full message history
    state = chat_graph.invoke({"messages": chat_history_langchain_format})

    # Extract the latest AI response
    ai_response = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break

    # Append new pair to history and return response + updated history
    history.append((message, ai_response))
    return ai_response










#GRADIO CHAT INTERFACE
chat_ui = gr.ChatInterface(
    fn=study_buddy,
    title="Your Study Buddy (LangGraph)",
    description="Ask me what you want to learn today!",
    examples=["Machine Learning", "Basic Python", "Data Structures"],
    theme="soft",
)


chat_ui.launch(share=False)