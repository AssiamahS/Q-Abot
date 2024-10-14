import os
import openai
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.doc_store import InMemoryDocStore
from pdfminer.high_level import extract_text

openai.api_key = 'your-openai-api-key'

# Load PDF and extract text
def load_pdf(file_path):
    text = extract_text(file_path)
    return text

# Create documents for DocStore
def create_documents_from_pdf(pdf_path):
    text = load_pdf(pdf_path)
    return [{"content": text, "title": "Manual"}]

# Initialize DocStore
docs = create_documents_from_pdf('your_document.pdf')
doc_store = InMemoryDocStore.from_documents(docs)

# Initialize LLM
llm = OpenAI(model_name='text-davinci-002')

# Create ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm_doc_store(llm=llm, doc_store=doc_store)

# Interaction example
def ask_question(question):
    response = chain.run(question)
    return response

# Example usage
response = ask_question("Where is the operating room?")
print(response)
