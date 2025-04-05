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

As a culmination of this learning journey, I will develop a **Virtual Study Buddy**. This application will leverage the concepts and tools explored during the course to provide an interactive and intelligent study companion.

### 🎯 Purpose:

An interactive AI tutor that helps users learn any topic by summarizing content, teaching with AI, quizzing them, and generating takeaway notes — all inside a Gradio UI.

---

### ✅ **Core Features to Include**

#### 1. **Topic Input**

- Gradio textbox: “What do you want to learn today?”
- Example: “Photosynthesis” or “Newton’s Laws”

#### 2. **Data Retrieval (RAG + Web Search)**

- Use Gemini to:
    - Search the web (or simulate it with predefined content if limited)
    - Summarize relevant information
    - Store it in memory/context

#### 3. **AI Agent Teaching Flow**

- Agent introduces the topic:
    
    *“Let’s learn about Photosynthesis! 🌱”*
    
- Shows a **short index** or bullet summary
- Waits for user to select a subtopic / say “Tell me more”

#### 4. **Follow-up Teaching**

- Responds with deeper explanation using Few-shot prompting or examples
- Optionally use Embedding to fetch the most relevant snippet (if you embed some docs in advance)

#### 5. **Quiz with Emojis 🧩**

- Ask 2–3 quiz questions (MCQ or True/False)
- Example: *“Which gas is absorbed during photosynthesis? 🌬️”*
- Reward right answers with 🎉, wrong with 🤔

#### 6. **PDF Notes Export 📄**

- Create a summary of what was learned
- Save as downloadable PDF using Python/Gradio tools

#### 7. **Motivational Outro 💪**

- “Great job! 🎓 Keep learning!”
- Suggest another topic or related content

Stay tuned for updates as I progress through this exciting learning experience!
