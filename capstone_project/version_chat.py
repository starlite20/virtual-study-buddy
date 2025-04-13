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
from langchain_core.messages import SystemMessage


from typing import Literal

from pprint import pprint






#State representing the learner's interaction with the study buddy
class StudyBuddyState(TypedDict):
    messages: list[BaseMessage]  # Remove Annotated wrapper
    focus_topics: list[str]
    finished: bool



# Assigning Role to the AI Agent
STUDYBUDDY_SYSINT =  SystemMessage(content="You are a helpful and encouraging Virtual Teacher. A student will share a topic they want to learn and you respond back with a very short summary (under 20 words), and provide a numbered list of 3-5 key subtopics that someone learning about the main topic should explore. Also, provide a short title for this learning activity (maximum 5 words).\n\nYour goal is to support their learning process by making all explanations very simplified and easy to understand. When the student asks a follow-up question, use the context of the previous conversation to provide a relevant and helpful answer.\n\nYou can help the learner by:\n- Answering their questions about the topics they are studying.\n- Providing clear and concise explanations of concepts.\n- Suggesting relevant learning resources like websites, articles, or videos (mentioning that you don't have direct access but can suggest where to look).\n- Helping them break down complex topics into smaller, more manageable parts (the initial subtopics).\n- Asking clarifying questions to ensure you understand their needs.\n- Offering encouragement and positive feedback.\n\n\nWhen the student initially provides a topic, your response should be structured to include a title, a brief summary, and a numbered list of subtopics. You will provide this information based on your understanding of the topic. The student may then ask questions about the main topic or any of the subtopics.\n\nIf you are unsure about something, admit that you don't know and suggest ways the learner might find the information (e.g., 'That's a great question! Let's try searching for it online' or 'Perhaps checking a textbook on that subject would be helpful.').\n\nOnce the learner indicates they are finished studying or want to end the session, acknowledge their decision, offer encouragement for their continued learning, and say goodbye!\n\nFeatures for managing study topics and generating practice questions are planned for future development.")

WELCOME_MSG = "Hi there! I'm your Virtual Study Buddy. What would you like to learn about today?"





#AI Model Initialization for LangGraph
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")



def chatbot_welcome(state: StudyBuddyState) -> StudyBuddyState:
    # Only add system message once
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        messages = [STUDYBUDDY_SYSINT] + state["messages"]
    else:
        messages = state["messages"]
    
    new_output = llm.invoke(messages)


    return {"messages": state["messages"] + [new_output]}


# Defining the study_buddy_bot
def chatbot(state: StudyBuddyState) -> StudyBuddyState:
    message_history = [STUDYBUDDY_SYSINT] + [msg for msg in state["messages"] if getattr(msg, "content", "").strip()]
    
    if not message_history:
        raise ValueError("Empty message history. Cannot invoke LLM.")

    new_output = llm.invoke(message_history)
    return {"messages": state["messages"] + [new_output]}


#user input
# human_node function
def human_node(state: StudyBuddyState) -> StudyBuddyState:
    # Get the ACTUAL last user message (HumanMessage)
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    print("User Messages:", user_messages)  # Debugging line
    if not user_messages:
        return state
    
    last_user_msg = user_messages[-1]
    
    # Check for exit command
    if last_user_msg.content.lower() in {"q", "quit", "exit", "goodbye"}:
        return {"finished": True}
    
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

graph_builder.add_edge(START, "human")  # Start with user input
graph_builder.add_edge("human", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    lambda state: END if state.get("finished") else "human"
)
graph_builder.add_edge("chatbot", END)

#graph_builder.add_edge("chatbot", END)


chat_graph = graph_builder.compile()

conversation_history = []

print("okay")








config = {"recursion_limit": 100}

def study_buddy(message: str, history: list) -> str:  
    print("Gradio history:", history)
    print("User message:", message)
    
    try:
        global chat_graph

        # Convert Gradio history to LangChain format
        chat_history = []
        for user_msg, bot_msg in history:
            chat_history.append(HumanMessage(content=user_msg))
            chat_history.append(AIMessage(content=bot_msg))
        
        # Add new user message
        chat_history.append(HumanMessage(content=message))
        
        # Create initial state
        initial_state = {
            "messages": chat_history,
            "focus_topics": [],
            "finished": False
        }

        
        print("Message history:")
        for msg in chat_history:
            print(f"{type(msg).__name__}: {repr(msg.content)}")
        # Run through LangGraph
        state = chat_graph.invoke(initial_state)
        
        # Extract AI response (last AIMessage)
        ai_response = next(
            msg.content 
            for msg in reversed(state["messages"]) 
            if isinstance(msg, AIMessage)
        )
        
        return ai_response
    
    except Exception as e:
        print("Exception occurred:")
        return f"Error: {str(e)}"









#GRADIO CHAT INTERFACE
chat_ui = gr.ChatInterface(
    fn=study_buddy,
    title="Your Study Buddy (LangGraph)",
    description="Ask me what you want to learn today!",
    examples=["Machine Learning", "Basic Python", "Data Structures"],
    theme="soft",
)


chat_ui.launch(share=False)