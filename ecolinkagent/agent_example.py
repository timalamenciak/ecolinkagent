from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os
from typing import List, Tuple
from oaklib import get_adapter
from oaklib.interfaces import TextAnnotatorInterface
# import gradio as gr

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))

annotate_agent = Agent(  
    model,
    deps_type=str,
    output_type=str,
    system_prompt="""
        You are an expert ecologist and data curator.
        Use your knowledge of ontologies to provide annotations for the input.
        Use terms from the local ontology file "elmo.owl" which can be accessed through the tool annotate_with_local_ontology to annotate the text
        Make sure that you also search the ontology file for synonyms to key terms
        If the local ontology file does not produce adequate results, use the search_ontology tool to search other ontologies like ENVO
    """
)


@annotate_agent.tool
async def annotate_with_local_ontology(ctx: RunContext[str], text, ontology_file="elmo.owl"):
    """
    Annotate text using a local ontology file with OAK.
    
    Args:
        text (str): The text to annotate
        ontology_file (str): Path to a local ontology file
    
    Returns:
        list: List of annotation results
    """
    try:
        # Create adapter for the local ontology file
        adapter = get_adapter(f"pronto:{ontology_file}")
        results = adapter.basic_search(text)
        labels = list(adapter.labels(results))

        # Display results
        print(f"Found {len(labels)} annotations: {labels}")
        
        return labels
        
    except Exception as e:
        print(f"Error processing ontology: {e}")
        return []
    

@annotate_agent.tool
async def search_ontology(ctx: RunContext[str], term: str, ontology: str, n: int, verbose: bool = False) -> List[Tuple[str, str]]:
    """
    Search an OBO ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.
        ontology: The ontology ID to search
        n: The number of results to return.

    Returns:
        A list of tuples, each containing an ontology ID and a label.
    """
    adapter = get_adapter(f"ols:{ontology}")
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))

    print(f"## TOOL USE: Searched for '{term}' in '{ontology}' ontology")
    print(f"## RESULTS: {labels}")
    return labels

# Run the agent
result = annotate_agent.run_sync('We planted 200 trees using a tree planting process in a farm field to try to restore a forest.')
print(result.output)