from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os

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
    system_prompt=(
        'You are an expert ecologist and data curator.'
        'Use your knowledge of ontologies to provide annotations for the input.'
        'Use terms from the local ontology file "elmo.owl" which can be accessed through the tool annotate_with_local_ontology to annotate the text'
        'Make sure that you also search the ontology file for synonyms to key terms'
    )
)


@annotate_agent.tool
async def roulette_wheel(ctx: RunContext[str], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'

@annotate_agent.tool
async def annotate_with_local_ontology(text, ontology_file="elmo.owl"):
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
        
        # Check if the adapter supports text annotation
        if not isinstance(adapter, TextAnnotatorInterface):
            print(f"Adapter does not support text annotation")
            return []
        
        # Perform annotation
        annotations = list(adapter.annotate_text(text))
        
        # Display results
        print(f"Found {len(annotations)} annotations:")
        for annotation in annotations:
            print(f"- Term: {annotation.object_id}")
            print(f"  Label: {annotation.object_label}")
            print(f"  Match: '{annotation.match_string}'")
            print()
        
        return annotations
        
    except Exception as e:
        print(f"Error processing ontology: {e}")
        return []

# Run the agent
result = annotate_agent.run_sync('We planted 200 trees using a tree planting process in a farm field to try to restore a forest.')
print(result.output)