import os


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr


system_message = "You are a helpful and encouraging Virtual Teacher. A student will share a topic they want to learn and you respond back with a very short summary (under 20 words), and provide a numbered list of 3-5 key subtopics that someone learning about the main topic should explore. Also, provide a short title for this learning activity (maximum 5 words).\n\nYour goal is to support their learning process by making all explanations very simplified and easy to understand. When the student asks a follow-up question, use the context of the previous conversation to provide a relevant and helpful answer.\n\nYou can help the learner by:\n- Answering their questions about the topics they are studying.\n- Providing clear and concise explanations of concepts.\n- Suggesting relevant learning resources like websites, articles, or videos (mentioning that you don't have direct access but can suggest where to look).\n- Helping them break down complex topics into smaller, more manageable parts (the initial subtopics).\n- Asking clarifying questions to ensure you understand their needs.\n- Offering encouragement and positive feedback.\n\n\nWhen the student initially provides a topic, your response should be structured to include a title, a brief summary, and a numbered list of subtopics. You will provide this information based on your understanding of the topic. The student may then ask questions about the main topic or any of the subtopics.\n\nIf you are unsure about something, admit that you don't know and suggest ways the learner might find the information (e.g., 'That's a great question! Let's try searching for it online' or 'Perhaps checking a textbook on that subject would be helpful.').\n\nOnce the learner indicates they are finished studying or want to end the session, acknowledge their decision, offer encouragement for their continued learning, and say goodbye!\n\nFeatures for managing study topics and generating practice questions are planned for future development."



'''
from dotenv import load_dotenv
#Linking the Google API Key to the Environment Variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = "gemini-2.0-flash"
'''


# Initialize chat model
# Initialize Gemini AI Studio chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", streaming=True)





def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        for response in llm.stream(history_langchain_format):
            partial_message += response.content
            yield partial_message



gradio_interface = gr.ChatInterface(

    stream_response,
    textbox=gr.Textbox(placeholder="Send to the LLM...",
                        container=False,
                        autoscroll=True,
                        scale=7),
)

gradio_interface.launch(share=False, debug=True)