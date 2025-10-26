import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_MODEL_API_VERSION")
#base_url="https://arune-mfpva4eo.openai.azure.com/openai/v1/",
client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://arune-mfpva4eo-eastus2.openai.azure.com/openai/v1",
)

response= client.responses.create(
        model="gpt-5-codex",  # Or "gpt-5" if you prefer the general-purpose model
        input="Write a Python function to calculate the factorial of a number recursively.",
        reasoning={"effort": "medium"},  # Adjust effort based on task complexity
        text={"verbosity": "medium"},  # Adjust verbosity of the output
    )
#response = client.chat.completions.create(
#    model="gpt-5-codex", # replace with the model deployment name of your o1 deployment.
#    messages=[
#        {"role": "user", "content": "What steps should I think about when writing my first Python API?"},
#    ],
#    max_completion_tokens = 5000
#
#)

print(response.model_dump_json(indent=2))
