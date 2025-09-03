from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import json

def create_template(input_variables, template):
    """
    This function creates the prompt template used to prompt the LLM model
    Arguments:
    input_variables: list -> The input variables used in the prompt
    templates: str -> The template
    """
    return PromptTemplate(
        input_variables=input_variables,
        template=template)

def get_response_from_llm(prompt_template, model, input_vars_dict):
    """
    This function prompts the LLM and gets a structures JSON output
    Arguments:
    prompt_template: str -> The prompt to use
    model: str -> The LLM to use
    input_vars_dict: dict -> Input Variables (as per PromptTemplate invoke method)
    """
    llm = ChatGoogleGenerativeAI(model=model)

    prompt = prompt_template.invoke(input_vars_dict)
    response = llm.invoke(prompt)
    
    raw = response.content.strip()
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        json_string = match.group(1)
        json_response = json.loads(json_string)
    return json_response

def dump_to_dot_json(file_name, json_response):
    with open(file_name, 'w') as f:
        json.dump(json_response, f)