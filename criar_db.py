from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#caminhos dos pdfs
caminhos = [
    "files/apostila.pdf",
    "files/LLM.pdf",
]

#carregar pag
paginas = []

for caminho in caminhos:
    loader = PyPDFLoader(caminho)

    paginas.extend(loader.load())

#splitter- divido em tamanho menor
recur_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(paginas)

#metadata
for i, doc in enumerate(documents):

    doc.metadata["source"] = (
        doc.metadata["source"]
        .replace("files/", "")
    )

    doc.metadata["doc_id"] = i

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#pasta do chroma
diretorio = "db/chat_retrieval"

#criar banco vetorial
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=diretorio
)

print("Banco vetorial criado com sucesso!")
print(f"Total de chunks: {len(documents)}")