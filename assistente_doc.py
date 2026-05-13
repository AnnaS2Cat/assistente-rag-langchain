from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(   #busca o significado e não só palavra =
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#pasta do banco vetorial
diretorio = "db/chat_retrieval"

#carrega chromadb
vectordb = Chroma(
    persist_directory=diretorio,
    embedding_function=embeddings_model
)

chat = ChatGroq(
    model="llama-3.1-8b-instant"
)

chain_prompt = PromptTemplate.from_template(
"""
Responda usando APENAS o contexto fornecido.
Se a resposta não estiver no contexto,
diga exatamente:
"Não encontrei essa informação nos documentos."
Não invente respostas.
Use no máximo 3 frases.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
)

#retrieval qa-mecanismo de busca de textos relevantes
chat_chain = RetrievalQA.from_chain_type(
    llm=chat,

    retriever=vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    ),

    chain_type="stuff",

    chain_type_kwargs={
        "prompt": chain_prompt
    },

    return_source_documents=True
)

#loop perguntas
while True:

    pergunta = input("\nPergunta: ")

    # sair
    if pergunta.lower() == "sair":
        break

    # resposta
    resposta = chat_chain.invoke({
        "query": pergunta
    })

    # mostrar resposta
    print("\nResposta:")
    print(resposta["result"])

    # mostrar fontes
    print("\nFontes utilizadas:")

    if len(resposta["source_documents"]) == 0:
        print("Nenhuma fonte encontrada.")

    else:

        for doc in resposta["source_documents"]:

            print(f"\nArquivo: {doc.metadata['source']}")
            print(f"Página: {doc.metadata['page']}")

            print("\nTrecho:")
            print(doc.page_content[:300])

            print("\n" + "-" * 50)