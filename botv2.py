import openai 
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm 
import os

# os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.environ.get('API_KEY', 'API key not found')

print("OPENAI_API_KEY Working...")

with open('content/Nikiya_Simon_Audio_otter_ai.txt', 'r') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int: 
    return len(tokenizer.encode(text, truncation=True, max_length=1024))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 0, 
    separators=["\n\nSimon Cullen"],
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

print(f"Split text into {len(chunks)} chunks...")

# helper function to grab previous 10 lines of output 
def lastFewLines(s) -> str: 
    count = 0
    i = len(s)-1
    
    while (i > 0 and count < 10):
        if (s[i] == '\n'): count += 1
        i -= 1 
    
    return s[i:len(text)]


output = []


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
    the side of preserving them. 

    Here is the first chunk of the source text:
    
    {chunks[0].page_content}
"""

# manually running the first prompt since it is slightly different 
completions0 = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": init_prompt}
    ]
)
message0 = completions0['choices'][0]['message']['content']
output.append(message0)

# running the rest of the chunks with this prompt 
for i in tqdm (range(1, len(chunks)), desc="Loading..."): 
    mid_prompt = f"""INSTRUCTIONS: "You are to create an edited transcript from 
        the text below. The text is from an interview between an RBT and an 
        interviewer, the interviewer will be named "Interviewer" and the RBT 
        will be names "RBT". Your editing should remove false starts, eliminate 
        filler words, and correct grammatical errors. But you should preserve 
        the details and keep the speakers' meaning intact. 

        Preserve all interesting details. If the RBT uses evocative language, 
        include the evocative language; never edit the final transcript to be less 
        evocative. If you are unsure about whether to eliminate some words, err on 
        the side of preserving them.   

        Here are the last 10 lines of the previous section for context: 

        {lastFewLines(output[i-1])}

        Here is the text that I want you to transcribe:

        {chunks[i].page_content}
    """

    completions = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": mid_prompt}
        ]
    )
    message = completions['choices'][0]['message']['content']
    output.append(message)

# writing to a new file 
f = open("outputs/outputv2.txt", "w")
for i in range(len(output)):
    f.write(output[i])
    f.write('\n\n')
f.close()

print("Completed writing to Output")