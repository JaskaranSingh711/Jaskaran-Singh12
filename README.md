from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def detect_emotion(text):
    result = emotion_model(text)
    return result[0]['label']
    from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
def generate_response(user_input, emotion):
    prompt = f"""
    User emotion: {emotion}
    
    If user is sad → be supportive
    If angry → be calm and understanding
    If happy → be energetic
    
    User: {user_input}
    AI:
    """
    return llm.predict(prompt)
    from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
while True:
    user_input = input("You: ")

    emotion = detect_emotion(user_input)
    response = generate_response(user_input, emotion)

    print(f"(Detected emotion: {emotion})")
    print("AI:", response)
    emotion_history.append(emotion)
