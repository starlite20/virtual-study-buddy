
from google import genai
from google.genai import types

from IPython.display import HTML, Markdown, display



import os
from dotenv import load_dotenv

#Linking the Google API Key to the Environment Variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = "gemini-2.0-flash"



#Retry Policy for Generative AI
from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)



'''
chat = client.chats.create(model=genai_model, history=[])
response = chat.send_message('Hello! My name is Zlork.')
print(response.text)

response = chat.send_message('Can you tell me something interesting about dinosaurs?')
print(response.text)

response = chat.send_message('Do you remember what my name is?')
print(response.text)
'''

system_message = "You are a helpful and encouraging Virtual Teacher. A student will share a topic they want to learn and you respond back with a very short summary (under 20 words), and provide a numbered list of 3-5 key subtopics that someone learning about the main topic should explore. Also, provide a short title for this learning activity (maximum 5 words).\n\nYour goal is to support their learning process by making all explanations very simplified and easy to understand. When the student asks a follow-up question, use the context of the previous conversation to provide a relevant and helpful answer.\n\nYou can help the learner by:\n- Answering their questions about the topics they are studying.\n- Providing clear and concise explanations of concepts.\n- Suggesting relevant learning resources like websites, articles, or videos (mentioning that you don't have direct access but can suggest where to look).\n- Helping them break down complex topics into smaller, more manageable parts (the initial subtopics).\n- Asking clarifying questions to ensure you understand their needs.\n- Offering encouragement and positive feedback.\n\n\nWhen the student initially provides a topic, your response should be structured to include a title, a brief summary, and a numbered list of subtopics. You will provide this information based on your understanding of the topic. The student may then ask questions about the main topic or any of the subtopics.\n\nIf you are unsure about something, admit that you don't know and suggest ways the learner might find the information (e.g., 'That's a great question! Let's try searching for it online' or 'Perhaps checking a textbook on that subject would be helpful.').\n\nOnce the learner indicates they are finished studying or want to end the session, acknowledge their decision, offer encouragement for their continued learning, and say goodbye!\n\nFeatures for managing study topics and generating practice questions are planned for future development."

WELCOME_MSG = "Hi there! I'm your Virtual Study Buddy. What would you like to learn about today?\n"




# Search grounding enabled.
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    system_instruction=system_message,
)

client = genai.Client(api_key=GOOGLE_API_KEY)
chat = client.chats.create(
    model=genai_model,
    config=config_with_search,
    )

response = chat.send_message(system_message)




def query_with_grounding(user_prompt):
    response = client.models.generate_content(
        model=genai_model,
        contents=user_prompt,
        config=config_with_search,
    )
    return response.candidates[0]


user_prompt = input(WELCOME_MSG + " -> ")
while user_prompt != "exit":
    response_chat = query_with_grounding(user_prompt)
    print (response_chat.content.parts[0].text)
    
    user_prompt = input(" -> ")

