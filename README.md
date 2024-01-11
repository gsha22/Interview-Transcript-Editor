# Trancription Editor with GPT-4

This Python script utilizes OpenAI's GPT-4 and Hugging Face's Transformers library to create an automated transcript editor. The tool is designed to process interview transcripts, eliminating false starts, filler words, and correcting grammatical errors while preserving the essence and details of the conversation.

## Background

I worked with Professor Simon Cullen from Carnegie Mellon University to help address an issue pertaining to interview transcripts: How can we use a large language model to our advantage to automatically edit long text files, and in doing so save hours of time? 

Because GPT-4 had recently come out, he advised me to find a way to create a script that can "read" interview text files and edit them in a way that the user pleases. 

## Features

- **Text Splitter:** Utilizes a Recursive Character Text Splitter to break down the input transcript into manageable chunks for efficient processing.

- **User Instructions:** Provides specific instructions to the model, guiding it in the editing process. The instructions include guidelines to retain evocative language and interesting details.

- **Dynamic Prompting:** Adapts the prompt based on the context by incorporating the last 10 lines of the previous section, ensuring continuity and coherence in the edited transcript.

- **Output Handling:** Writes the edited transcript to a new file in the "outputs" folder, making it easily accessible for further analysis.

## Dependencies 

- OpenAI's GPT-4
- HuggingFace Transformers
- tqdm
- langchain

## Usage

1. Ensure you have the necessary dependencies installed.

2. Create an environ key in your os for your OpenAI API key.

3. Place your interview transcripts in the "content" directory.

4. Make any edits you wish to the "init_prompt" and/or "mid_prompt" to change how GPT-4 will edit your txt file. 

5. Run the script, and the edited transcripts will be generated and stored in the "outputs" folder.

