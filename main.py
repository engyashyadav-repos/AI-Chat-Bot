from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a  helpful AI assistant."
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{question}"
    )
])

chat_History = []
while True:

    user = input("You: ")
    if user.strip().lower() in ["bye", "exit", "quit"]:
        print("AI: Bye")
        break

    
    prompt_Response = prompt.invoke({

        "history": chat_History,
        "question": user
        
    })
    chat_History.append(HumanMessage(content=user))
    

    print("AI: ",end="")
    response = model.invoke(prompt_Response)
        
    print(response.content, end="", flush=True)
    print("\n")
    chat_History.append(AIMessage(content=response.content))

