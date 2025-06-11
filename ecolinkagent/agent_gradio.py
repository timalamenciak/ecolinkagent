from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os
import asyncio
from oaklib import get_adapter
from oaklib.interfaces import TextAnnotatorInterface
import gradio as gr
import rdflib
from rdflib import URIRef, BNode, Literal
from rdflib import Namespace
from rdflib.namespace import DC, DCAT, DCTERMS, FOAF, OWL, PROF, RDF, RDFS, SKOS, TIME, XSD


load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))

annotate_agent = Agent(  
    model,
    deps_type=str,
    output_type=str,
    system_prompt=(
        'You are an expert ecologist and data curator. '
        'Use your knowledge of ontologies to provide annotations for the input. '
        'Use terms from the local ontology file "elmo.owl" which can be accessed through the tool annotate_with_local_ontology to annotate the text. '
        'Make sure that you also search the ontology file for synonyms to key terms. '
        'Provide clear explanations of the annotations you find and how they relate to the ecological concepts in the text.'
        'Please output each annotation in accurate RDF - e.g. subject - predicate - object'
    )
)

@annotate_agent.tool
async def search_ontology(term: str, ontology: str, n: int, verbose: bool = False) -> List[Tuple[str, str]]:
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
    adapter = get_adapter("ols:{ontology}")
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))

    print(f"## TOOL USE: Searched for '{term}' in '{ontology}' ontology")
    print(f"## RESULTS: {labels}")
    return labels

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
        results = adapter.basic_search(text)
        labels = list(adapter.labels(results))

        # Display results
        print(f"Found {len(labels)} annotations: {labels}")
        
        return labels
        
    except Exception as e:
        print(f"Error processing ontology: {e}")
        return []        


def process_message(message, history):
    """Process a message through the annotation agent"""
    try:
        # Run the agent synchronously
        result = annotate_agent.run_sync(message, deps="")
        return result.output
    except Exception as e:
        return f"Error processing message: {str(e)}"

# Create the Gradio interface
def create_chat_interface():
    """Create and launch the Gradio chat interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px !important;
    }
    """
    
    with gr.Blocks(css=css, title="Ecological Annotation Agent") as demo:
        gr.Markdown(
            """
            # 🌱 Ecological Annotation Agent
            
            This AI agent specializes in ecological data curation and ontology-based text annotation.
            It uses the ELMO ontology to identify and annotate ecological terms in your text.
            
            **Try asking about:**
            - Forest restoration processes
            - Plant ecology terminology  
            - Environmental conservation methods
            - Species interactions and habitats
            """
        )
        
        chatbot = gr.Chatbot(
            height=500,
            placeholder="Ask me to annotate ecological text or explain ecological concepts...",
            show_label=False
        )
        
        msg = gr.Textbox(
            placeholder="Enter your ecological text for annotation...",
            show_label=False,
            container=False,
            scale=7
        )
        
        with gr.Row():
            submit = gr.Button("Submit", variant="primary", scale=1)
            clear = gr.Button("Clear Chat", scale=1)
        
        gr.Examples(
            examples=[
                "We planted 200 trees using a tree planting process in a farm field to try to restore a forest.",
                "The mycorrhizal fungi formed symbiotic relationships with the tree roots in the deciduous woodland ecosystem.",
                "Carbon sequestration occurs when trees absorb CO2 from the atmosphere during photosynthesis.",
                "The biodiversity hotspot contained multiple endemic species adapted to the mediterranean climate."
            ],
            inputs=msg,
            label="Example queries:"
        )
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            # Process the message
            bot_message = process_message(message, chat_history)
            
            # Add to chat history
            chat_history.append((message, bot_message))
            return chat_history, ""
        
        def clear_chat():
            return [], ""
        
        # Event handlers
        submit.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(clear_chat, None, [chatbot, msg])
    
    return demo

if __name__ == "__main__":
    # Test the agent with the original example
    print("Testing agent with example:")
    result = annotate_agent.run_sync('We planted 200 trees using a tree planting process in a farm field to try to restore a forest.')
    print("Agent result:", result.output)
    print("\n" + "="*50 + "\n")
    
    # Launch the Gradio interface
    print("Launching Gradio chat interface...")
    demo = create_chat_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=True
    )