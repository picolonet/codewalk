import yaml
import os
import time
from llm_providers import GroqProvider, OpenAIProvider, ClaudeProvider

# --- Configuration ---
# Define the models you want to query.
# NOTE: GPT-5 and Llama 4 are not yet released.
# I am using the latest available models as placeholders.
# You can update the model names here when they become available.
LLM_MODELS = {
    "groq_llama4": GroqProvider(model="meta-llama/llama-4-maverick-17b-128e-instruct"),
    "groq_llama3": GroqProvider(model="llama-3.3-70b-versatile"),
    # "groq_llama4": GroqProvider(model="llama4-placeholder"), # Placeholder for Llama 4
    "openai_gpt4o": OpenAIProvider(model="gpt-5"), # Placeholder for GPT-5
    "claude_sonnet": ClaudeProvider(model="claude-sonnet-4-20250514"),
}

INPUT_YAML_FILE = 'dj_qa.yaml'
OUTPUT_YAML_FILE = 'dj_qa_answered.yaml'

def process_questions():
    """
    Reads questions from a YAML file, gets answers from configured LLM providers,
    and saves the results to a new YAML file.
    """
    print(f"Loading questions from '{INPUT_YAML_FILE}'...")
    try:
        with open(INPUT_YAML_FILE, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_YAML_FILE}' was not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    if 'questions' not in data or not isinstance(data['questions'], list):
        print("Error: YAML file must contain a list of questions under the 'questions' key.")
        return

    questions = data.get('questions', [])
    total_questions = len(questions)
    print(f"Found {total_questions} questions to process.")

    for i, item in enumerate(questions):
        question_text = item.get('question')
        if not question_text:
            print(f"Skipping item {i+1} as it has no 'question' field.")
            continue

        print(f"\n--- Processing Question {i+1}/{total_questions} ---")
        print(f"Q: {question_text[:100]}...")

        for model_label, provider in LLM_MODELS.items():
            answer_key = f"{model_label}_answer"

            if answer_key in item and item[answer_key]:
                print(f"-> Answer from '{model_label}' already exists. Skipping.")
                continue

            print(f"--> Querying '{model_label}'...")
            try:
                answer = provider.get_answer(question_text)
                item[answer_key] = answer
                print(f"--> Received answer from '{model_label}'.")
            except Exception as e:
                print(f"--> ERROR: Could not get answer from '{model_label}': {e}")
                item[answer_key] = f"Error: {e}"
            
            # Add a small delay to avoid hitting rate limits too quickly
            time.sleep(1)

    print("\n--- Processing complete ---")

    # Write the updated data to the output file
    try:
        with open(OUTPUT_YAML_FILE, 'w') as f:
            yaml.dump(data, f, sort_keys=False, width=100, allow_unicode=True)
        print(f"Successfully saved answers to '{OUTPUT_YAML_FILE}'.")
    except Exception as e:
        print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    process_questions()

