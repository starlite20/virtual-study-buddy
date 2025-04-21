# kaggle-intensive-genai

This project is a showcase of my journey in learning and implementing Generative AI concepts as part of the 5-Day Intensive Course offered by Google and Kaggle.

## Overview

Throughout this project, I will explore various concepts and techniques in Generative AI, including but not limited to:

- Structured output/JSON mode/controlled generation
- Few-shot prompting
- Document understanding
- Image understanding
- Video understanding
- Audio understanding
- Function calling
- Agents
- Long context window
- Context caching
- Gen AI evaluation
- Grounding
- Embeddings
- Retrieval augmented generation (RAG)
- Vector search/vector store/vector database
- MLOps (with GenAI)

The primary model used for this project will be **Gemini Flash 2.0**.

## Capstone Project

As a culmination of this learning journey, I will develop **Study Buddy**, GenAI-Powered Learning Assistant. This application is an AI-powered chatbot designed to simplify the learning process for users by teaching complex concepts as simplified with beginner-friendly explanations of any topic the user wants to explore. Built using Generative AI, and LangGraph, it creates an interactive chat-based interface where learners can engage in a conversational learning experience.

The agent listens to the user’s query and responds with an encouraging tone, breaking down the topic into a short summary, a brief title, and a list of subtopics to dive deeper. The bot adapts based on the user’s inputs and can leverage external tools via function calling to enhance the learning support.



### 🎯 Purpose:

Traditional learning platforms often overwhelm users with too much complexity and non-interactive formats. Beginners usually get overwhelmed with the amount of information they need to swallow. They will benefit from **human-like conversation**, **personalized guidance**, and **bite-sized explanations**.

**Virtual Study Buddy** addresses this gap by:
- Using GenAI to **simulate a human tutor**.
- Responding with **clear, simplified, and relevant content**.
- Making the learning journey feel **personal and manageable**.

This is particularly valuable for:
- Self-learners  
- Students preparing for exams


---

### ✅ **Core Features to Include**
## How GenAI Features are embedded into this project?

#### -> AI Agents
LangGraph’s agent architecture enables **modular design**, allowing state transitions and logic to evolve in response to user interaction. This is perfect for handling a learning flow where the AI adapts to new topics or user intentions dynamically.

#### -> Function Calling (Tool Usage)
Instead of coding rigid topic trees, this system uses **function calling (tool usage)** to dynamically plug in expert functions when needed — enabling real-time topic resolution or content generation tailored to the student’s needs. We are utilizing the Wikipedia Tool here.

This allows:
- Scalability with extension with multiple Tools 
- Dynamic topic-specific behavior  

#### -> Few-Shot Prompting
Few-shot prompting powers the initial learning behavior. It sets expectations by providing examples of how to:
- Respond with summaries  
- List subtopics  
- Keep explanations friendly and digestible  

This eliminates the need for custom training or RAG, while keeping the system behavior consistent and easy to adapt.

#### -> Structured Output
The Output from the AI Agent is very clear about which is the Human Message, which is the AI Message, and which is the Tool Message. Therefore the Output shared is shared in a clear segregated manner.




## Conclusion & Future Expansions
This project marks the foundational step toward building an AI-powered educational companion. By leveraging GenAI capabilities such as AI agents, function calling, and few-shot prompting, it demonstrates how conversational interfaces can simplify complex topics and personalize the learning experience.

### Potential Future Enhancements:
- **RAG Integration**: Incorporate Retrieval-Augmented Generation to fetch up-to-date content from external knowledge bases or curated learning materials.
- **Tool Enrichment**: Add specialized tools for different domains (e.g., code interpreters, math solvers, or visual explainer modules).
- **Contextual Memory**: Enhance session continuity with memory capabilities to track user progress and adapt future interactions accordingly.
- **Multimodal Support**: Extend capabilities to include images, diagrams, or video recommendations for visual learners.
- **Gamified Learning**: Introduce quizzes, scoreboards, or achievements to keep learners motivated and engaged.
- **Note Export & Summaries**: Let users export their learning journey as structured notes or PDFs.
- **Gradio UI Improvements**: Upgrade the user interface with avatars, animations, and support for multi-turn conversations in a more natural flow.

With these additions, the Study Buddy can evolve from a simple topic explainer into a fully adaptive and engaging learning assistant for a wide range of users.

*Learning is just the beginning.
Let's build the future of education, one question at a time.*
