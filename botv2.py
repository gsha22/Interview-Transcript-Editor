import openai 
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm 
import os
from pathlib import Path

# helper function to grab previous 10 lines of output  
def lastFewLines(s) -> str: 
    count = 0
    i = len(s)-1
    
    while (i > 0 and count < 10):
        if (s[i] == '\n'): count += 1
        i -= 1 
    
    return s[i:len(s)]

def transcript_editor(file_path, interviewer_name): 
    with open(f'{file_path}', 'r') as f:
        text = f.read()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int: 
        return len(tokenizer.encode(text, truncation=True, max_length=3000))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 0, 
        separators=[f"\n\n{interviewer_name}"],
        length_function = count_tokens,
    )

    chunks = text_splitter.create_documents([text])

    print(f"Split text into {len(chunks)} chunks...")


    output = []


    init_prompt = f"""
        INSTRUCTIONS: ou are to create an edited transcript from the text below. 
        The text is from an interview between an RBT and an interviewer, the 
        interviewer will be named "Interviewer" and the RBT will be names "RBT". 
        Your editing should remove false starts, eliminate filler words, and 
        correct grammatical errors. 

        Preserve all interesting details. If the RBT uses evocative language, 
        include the evocative language; never edit the final transcript to be 
        less evocative. If you are unsure about whether to eliminate some words, 
        err on the side of preserving them. 

        Here is the first chunk of the source text:

        {chunks[0].page_content}
    """

    # running the rest of the chunks with this prompt 
    for i in tqdm (range(len(chunks)), desc="Loading..."): 
        if i == 0: 
            completions0 = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                messages=[
                    {"role": "user", "content": init_prompt}
                ]
            )
            message0 = completions0['choices'][0]['message']['content']
            output.append(message0)
        else: 
            mid_prompt = f"""INSTRUCTIONS: You are to create an edited transcript 
                from the text below. The text is from an interview between an RBT 
                and an interviewer, the interviewer will be named "Interviewer" and 
                the RBT will be names "RBT". Your editing should remove false 
                starts, eliminate filler words, and correct grammatical errors. 
                But you should preserve the details and keep the speakers' meaning 
                intact. Preserve all interesting details. If the RBT uses evocative 
                language, include the evocative language; never edit the final 
                transcript to be less evocative. If you are unsure about whether to 
                eliminate some words, err on the side of preserving them.

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
    file_name = os.path.basename(file_path)

    f = open(f"outputs/{file_name}", "w")
    for i in range(len(output)):
        f.write(output[i])
        f.write('\n\n')
    f.close()

openai.api_key = os.environ.get('API_KEY', 'API key not found')

print("OPENAI_API_KEY Working...\n")

interviewer_names = {"Brandon Zhou", "Ava Allard", "Jei Park", "Kyle", "Lily", "Rebecca", "Ricky", "Simon Cullen"}

def name_getter(file) -> str:
    f = open(f'{file}', 'r')
    lines = f.read().splitlines()
    for i in range(4):
        if len(lines[i]) > 13: continue 
        elif lines[i] in interviewer_names: 
            return lines[i].strip()
    return "Unknown Interviewer"

# iterate over files in
# that directory
files = Path('content').glob('*')
for file in files:
    print(f"Editing file: {os.path.basename(file)}")
    transcript_editor(file, name_getter(file))
    print(f"Finished editing file: {os.path.basename(file)}. You can now find it in the outputs folder.")
    print('\n')

print("Completed editing all files. You can find the edited version in the outputs folder.")