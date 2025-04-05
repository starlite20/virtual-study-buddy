# Define the expected JSON structure
class StudyData(BaseModel):
    summary: str
    subtopics: list[str]

def study_buddy(input_text):
    global current_topic
    if not hasattr(study_buddy, 'current_topic'):
        study_buddy.current_topic = None
        study_buddy.subtopics_list = [] # Initialize subtopics list

    response = ""

    if not study_buddy.current_topic:
        study_buddy.current_topic = input_text
        topic = study_buddy.current_topic

        # Prompt Gemini for summary and subtopics in JSON format
        prompt = f"""Summarize the key concepts of '{topic}' in a few sentences.
Then, suggest 2-3 key subtopics someone learning about '{topic}' should explore.
Return your response as a JSON object with the following structure:
{{
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
            search_summary = study_data.summary
            study_buddy.subtopics_list = study_data.subtopics

            subtopics_display = "\n".join([f"- {sub}" for sub in study_buddy.subtopics_list]) if study_buddy.subtopics_list else "No subtopics suggested yet."

            response = f"{topic}:\n\n{search_summary}\n\nHere are some subtopics we can explore:\n{subtopics_display}\n\nType a subtopic to learn more, or say 'Tell me more' to start with the first one."

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
    inputs=gr.Textbox(label="What do you want to learn today?"),
    outputs="text",
    title="Virtual Study Buddy",
    description="Enter a topic you'd like to learn about."
)

iface.launch(share=True)