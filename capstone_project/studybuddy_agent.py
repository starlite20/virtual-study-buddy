

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai_model = "gemini-2.0-flash"

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")





from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition



from typing import Annotated #Labelling
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage #Messages can be AI, Human or System
from langgraph.graph.message import add_messages #Reducer for Langgraph

from langchain_google_genai import ChatGoogleGenerativeAI



#Configuring the LLM
llm = ChatGoogleGenerativeAI(model=genai_model)
#test_prompt = "hi how are you?"
#response = llm.invoke(test_prompt)
#print(response)





#Configuring Wikipedia as a Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results = 1,doc_content_chars_max = 500)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_api_wrapper, description="Query Wikipedia.")
#print (wikipedia.invoke("quantum computing"))





#Compilation of all the Tools
#For this phase only one tool
tools = [wikipedia]



#Binding the Tools with the LLM
llm_with_tools = llm.bind_tools(tools=tools)
#print(llm_with_tools.invoke("what is electric power"))


#Creating the StateGraph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]




#Connecting the Nodes and the Tools
def tool_calling_llm(state:State):
    return { "messages":[llm_with_tools.invoke(state["messages"])]}

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



#Sample Call with Prompt
messages=main_graph.invoke({"messages":"quantum computer"})
for m in messages['messages']:
    m.pretty_print()







