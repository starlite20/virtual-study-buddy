##Tools Creation
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


#ARXIV
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results = 2,doc_content_chars_max = 500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers.")
print (arxiv.name)
#arxiv.invoke("quantum computing")



#WIKIPEDIA
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results = 1,doc_content_chars_max = 500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Query Wikipedia.")
print (wikipedia.name)
print (wikipedia.invoke("quantum computing"))









import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

