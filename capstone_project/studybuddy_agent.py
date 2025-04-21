import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai_model = "gemini-2.0-flash"



from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=genai_model)
test_prompt = "hi how are you?"
response = llm.invoke(test_prompt)
print(response)



from typing import Annotated #Labelling
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage #Messages can be AI, Human or System
from langgraph.graph.message import add_messages #Reducer for Langgraph






from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results = 1,doc_content_chars_max = 500)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_api_wrapper, description="Query Wikipedia.")
print (wikipedia.invoke("quantum computing"))



tools = [wikipedia]

llm_with_tools = llm.bind_tools(tools=tools)
print(llm_with_tools.invoke("what is electric power"))





#from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition





class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    learning_completed: bool
    exam_completed: bool


system_role = "You are a helpful and encouraging Virtual Teacher. A student will share a topic they want to learn and you respond back with a very short summary (under 20 words), and provide a numbered list of 3-5 key subtopics that someone learning about the main topic should explore. Also, provide a short title for this learning activity (maximum 5 words)."

welcome_message = "Welcome Learner! I'm your StudyBuddy, here to help you learn in the simplest manner possible. Share a topic you're interested in, and lets chillax and learn together."







def welcome(state:State):
    return welcome_message

def tool_calling_llm(state:State):
    return { "messages":[llm_with_tools.invoke(state["messages"])]}


def human_node(state:State) -> State:
    last_msg = messages["messages"][-1]
    print("AI Response : ", last_msg)

    user_input = input(" You : ")

    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}







graph = StateGraph(State)
graph.add_node("tool_calling_llm", tool_calling_llm)
graph.add_node("tools", ToolNode(tools))


graph.add_edge(START, "tool_calling_llm")
graph.add_conditional_edges(
    "tool_calling_llm",
    #if latest message calls tool
    #if latest message doesnt need tool it goes to end
    tools_condition,
)

graph.add_edge("tools", "tool_calling_llm")

main_graph = graph.compile()
#display(Image(main_graph.get_graph().draw_mermaid_png()))





messages=main_graph.invoke({"messages":"how are you?"})
for m in messages['messages']:
    m.pretty_print()





messages=main_graph.invoke({"messages":"quantum computer"})
for m in messages['messages']:
    m.pretty_print()