import os
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI 
from langchain.chains import ConversationalRetrievalChain 
import matplotlib.pyplot as plt
import time 


from IPython.display import display
import ipywidgets as widgets

os.environ["OPENAI_API_KEY"] = "{API_KEY}" 

with open('content/Nikiya_Simon_Audio_otter_ai.txt', 'r') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int: 
    return len(tokenizer.encode(text, truncation=True, max_length=512))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap = 0, 
    separators=["\n\nSimon Cullen"],
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

print(len(chunks))

init_prompt = f"""
    INSTRUCTIONS: You are to create an edited transcript from the text below. 
    The text is from an interview between an RBT and an interviewer, the 
    interviewer will be named "Interviewer" and the RBT will be names "RBT". 
    Your editing should remove false starts, eliminate filler words, and 
    correct grammatical errors. But you should preserve the details and keep 
    the speakers' meaning intact.

    Preserve all interesting details. If the RBT uses evocative language, 
    include the evocative language; never edit the final transcript to be less 
    evocative. If you are unsure about whether to eliminate some words, err on 
    the side of preserving them. Remove metadiscourse. 

    Here is the first chunk of the source text:
    
    {chunks[0]}
"""


output = []

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# initial part done manually 
doc0 = db.similarity_search(init_prompt)
output.append(chain.run(input_documents=doc0, question=init_prompt))


for i in range(1, len(chunks)): 
    print(f"{i} is working\n\n")
    mid_prompt = f"""Please continue processing the text using these directions: "You are to 
    create an edited transcript from the text below. The text is from an 
    interview between an RBT and an interviewer, the interviewer will be 
    named "Interviewer" and the RBT will be names "RBT". Your editing should 
    remove false starts, eliminate filler words, and correct grammatical 
    errors. But you should preserve the details and keep the speakers' meaning 
    intact. Preserve all interesting details. Remove metadiscourse. 

    Here is the text:

    {chunks[i]}
    """

    doc = db.similarity_search(mid_prompt)
    output.append(chain.run(input_documents=doc, question=mid_prompt))


f = open("outputs/output.txt", "w")
for i in range(len(output)):
    f.write(output[i])
    f.write('\n')
f.close()

print("Completed writing to Output")
######################################################################


# # Quick data visualization to ensure chunking was successful

# # Create a list of token counts
# token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# # Create a DataFrame from the token counts
# df = pd.DataFrame({'Token Count': token_counts})

# # Create a histogram of the token count distribution
# df.hist(bins=40, )

# # Show the plot
# plt.show()

##################################################################

# embeddings = OpenAIEmbeddings()

# db = FAISS.from_documents(chunks, embeddings)

# # query = "explain top k frequent elements"
# # docs = db.similarity_search(query)

# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# query = "Create a grammatically correct and readable transcript of the raw transcript given from start to finish. Do not summarize, and keep the integrity of the original work"
# docs = db.similarity_search(query)
# print(chain.run(input_documents=docs, question=query))


# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# chat_history = []

# def on_submit(_):
#     query = input_box.value
#     input_box.value = ""

#     if query.lower() == 'exit':
#         print("Thank you for using the State of the Union chatbot!")
#         return

#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))

#     display(widgets.HTML(f'<b>User:</b> {query}'))
#     display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

# print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

# input_box = widgets.Text(placeholder='Please enter your question:')
# input_box.on_submit(on_submit)

# display(input_box)