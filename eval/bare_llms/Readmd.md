LLM Q&A Processor
This project provides a Python script to automate the process of asking a set of questions from a YAML file to various Large Language Models (LLMs) and saving their answers back into the YAML file.

The system is designed to be modular, making it easy to add, remove, or update the LLM providers.

Features
Reads questions from a structured YAML file.

Queries multiple LLM APIs:

Groq (for Llama 3)

OpenAI (for GPT models)

Anthropic (for Claude models)

Saves the answers back into a new YAML file with appropriate labels.

Skips questions that already have answers, allowing you to resume processing.

Modular design to easily support more models in the future.

Setup Instructions
1. Prerequisites
Python 3.7 or higher

pip for installing packages

2. Clone the Repository
Clone or download the project files into a local directory.

3. Install Dependencies
Install the required Python packages using the requirements.txt file:

pip install -r requirements.txt

4. Set Up API Keys
The script requires API keys for the LLM services you want to use. It loads these keys from environment variables.

Create a file named .env in the root directory of the project and add your API keys in the following format:

GROQ_API_KEY="your_groq_api_key"
OPENAI_API_KEY="your_openai_api_key"
CLAUDE_API_KEY="your_anthropic_api_key"

Note: You only need to provide keys for the services you intend to use. If you don't provide a key for a service, the script will raise an error when it tries to initialize that provider. You can comment out unused providers in llm_qa_processor.py.

5. Prepare the Input File
Ensure your input YAML file is named dj_qa.yaml and is placed in the same directory as the script. The file should have a top-level key questions which contains a list of question objects. Each object must have a question key.

Example dj_qa.yaml structure:

title: My Question Set
questions:
  - question: What is the capital of France?
    reference_answer: |
      Paris is the capital of France.
  - question: Explain the theory of relativity.

How to Run
Execute the main script from your terminal:

python llm_qa_processor.py

The script will:

Read the questions from dj_qa.yaml.

Iterate through each question and send it to the configured LLMs.

Print progress to the console.

Save the original questions along with the new answers to dj_qa_answered.yaml.

Customization
Adding or Changing LLM Models
You can easily change the models or add new ones by editing the LLM_MODELS dictionary at the top of llm_qa_processor.py.

For example, when Llama 4 is released on Groq, you can uncomment and update the placeholder:

LLM_MODELS = {
    "groq_llama3": GroqProvider(model="llama3-70b-8192"),
    "groq_llama4": GroqProvider(model="llama4-new-model-name"), # Update the model name
    # ... other models
}

Adding a New LLM Provider
Open llm_providers.py.

Create a new class that inherits from LLMProvider.

Implement the get_answer(self, question: str) method using the new provider's API client.

Add the new provider to the LLM_MODELS dictionary in llm_qa_processor.py.
