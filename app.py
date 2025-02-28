import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import json
import os
import ast
from time import sleep
import re

# Set page config
st.set_page_config(
    page_title="MS365 Template Matching...",
    page_icon="📝",
    layout="wide"
)

# Get API keys from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to initialize Pinecone (updated for new Pinecone client)
def init_pinecone():
    # Initialize Pinecone client with the new syntax
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if the index exists
    existing_indexes = pc.list_indexes().names()
    if "ms365" not in existing_indexes:
        # Create index if it doesn't exist using the new ServerlessSpec
        pc.create_index(
            name="ms365",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )

    # Connect to the index - new way
    index = pc.Index("ms365")
    return index

# Function to extract relevant content from alerts
def extract_relevant_content(full_content):
    """
    Extract the relevant portion of the alert content - the description between DU number and Link to 365
    """
    try:
        # Look for pattern "DU = <number>" followed by alert content
        du_match = re.search(r'DU\s*=\s*\d+(.+?)(?:Link to 365|This user has)', full_content, re.DOTALL)

        if du_match:
            # Extract the text between DU and Link to 365
            return du_match.group(1).strip()
        else:
            # If pattern not found, return the original content
            return full_content
    except Exception as e:
        st.error(f"Error extracting relevant content: {str(e)}")
        return full_content

# Function to get embedding from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to analyze content with OpenAI LLM
def analyze_with_llm(content, templates):
    # Extract the relevant portion of the content
    relevant_content = extract_relevant_content(content)

    # Prepare a list of template titles and IDs for the LLM
    template_info = []
    for template in templates:
        template_info.append({
            "id": template.get("ID"),
            "title": template.get("title")
        })

    # Create a system prompt that explains the task
    system_prompt = """
    You are an AI assistant specialized in analyzing MS365 security and compliance content.
    Your task is to match the provided content with the most appropriate template from a list of templates.
    Analyze the content carefully and determine which template would be most suitable for this type of event or issue.
    Return ONLY the ID of the best matching template as a number without any additional text.
    """

    # Create a user prompt with the content and template information
    user_prompt = f"""
    Content to analyze:
    {relevant_content}
    
    Available templates:
    {json.dumps(template_info, indent=2)}
    
    Based on the content, which template ID is the best match? Respond with only the template ID as a number.
    """

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini as requested
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # Use zero temperature for more deterministic responses
            max_tokens=10   # We only need a short response (the ID)
        )

        # Extract the template ID from the response
        template_id_str = response.choices[0].message.content.strip()

        # Try to convert to an integer or return as is if it's not a number
        try:
            template_id = int(template_id_str)
            return template_id
        except ValueError:
            # If it's not a clean number, try to extract a number
            match = re.search(r'\d+', template_id_str)
            if match:
                return int(match.group())
            return template_id_str

    except Exception as e:
        st.error(f"Error analyzing content with LLM: {str(e)}")
        return None

# Function to query Pinecone for similar content
def query_pinecone(index, content):
    try:
        # Extract relevant content for analysis
        relevant_content = extract_relevant_content(content)

        # Convert content to vector
        vector = get_embedding(relevant_content)

        # Query Pinecone with the new client
        results = index.query(
            vector=vector,
            top_k=5,
            include_metadata=True
        )

        return results
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return None

# Function to get all templates from Pinecone
def get_all_templates_from_pinecone(index):
    try:
        # This is a simplification - in a real-world scenario, you would need to 
        # implement pagination if you have more than a few hundred templates
        # For demonstration, assuming we have fewer than 1000 templates
        # Note: Pinecone doesn't have a "get all" function, so this is an approximation

        # Use a dummy vector to fetch all records
        # This isn't ideal but works for demo purposes
        dummy_vector = [0.0] * 1536  # Match the dimension of your index

        results = index.query(
            vector=dummy_vector,
            top_k=1000,  # Adjust based on your expected number of templates
            include_metadata=True
        )

        templates = []
        for match in results.matches:
            template = {
                "ID": int(match.id) if match.id.isdigit() else match.id,
                "title": match.metadata.get("title", ""),
                "name": match.metadata.get("name", ""),
                "kind": match.metadata.get("kind", ""),
                "active": match.metadata.get("active", False),
                "template": match.metadata.get("template", "")
            }
            templates.append(template)

        return templates
    except Exception as e:
        st.error(f"Error fetching templates from Pinecone: {str(e)}")
        return []

# Function to display the template content in a user-friendly way
def display_template_content(template_content):
    # If the template is HTML-like, render it as HTML
    if template_content and ("<div" in template_content or "<p" in template_content):
        st.markdown("### Template Content")
        st.markdown("---")
        st.markdown(template_content, unsafe_allow_html=True)
    else:
        # Otherwise just display as text
        st.markdown("### Template Content")
        st.markdown("---")
        st.text(template_content)

# Function to check if required environment variables are set
def check_environment_variables():
    missing_vars = []
    if not PINECONE_API_KEY:
        missing_vars.append("PINECONE_API_KEY")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")

    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set these variables in your Streamlit Cloud secrets or .env file.")
        return False
    return True

# Main application
def main():
    st.title("MS365 Template Analysis Tool")

    # Check if environment variables are set
    if not check_environment_variables():
        return

    # Initialize Pinecone
    try:
        index = init_pinecone()
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return

    # Content input with simple example placeholder - placed before the button
    content = st.text_area("Paste your content here for analysis : (Pasting Content #m365-alerts Slack channel)", height=300,
                         placeholder="Example : \nUNUSUAL SIGN IN (FOREIGN_COUNTRY)\nDU = 1810\n{\n  ..\n  ..\n  ..\n}\nLink to 365")

    # Analyze button
    if st.button("Analyze Content"):
        if content:
            with st.spinner("Analyzing content..."):
                # Extract the relevant portion
                relevant_content = extract_relevant_content(content)

                # First approach: Query Pinecone for similar content using vector similarity
                pinecone_results = query_pinecone(index, content)

                # Second approach: Use LLM to analyze content and find matching template
                # Get all templates from Pinecone for LLM analysis
                all_templates = get_all_templates_from_pinecone(index)

                if all_templates:
                    llm_template_id = analyze_with_llm(content, all_templates)

                    # Find the template with the matching ID from LLM analysis
                    llm_matching_template = None
                    for template in all_templates:
                        if str(template.get("ID")) == str(llm_template_id):
                            llm_matching_template = template
                            break
                else:
                    llm_matching_template = None
                    llm_template_id = None

                # Determine which method to trust more (you can customize this logic)
                has_vector_match = (pinecone_results and hasattr(pinecone_results, 'matches') and 
                                   len(pinecone_results.matches) > 0)

                if has_vector_match and pinecone_results.matches[0].score > 0.8:
                    # If vector similarity is very high, trust that
                    final_match = {
                        "id": pinecone_results.matches[0].id,
                        "title": pinecone_results.matches[0].metadata.get('title'),
                        "method": "Content Pattern Match",
                        "template_content": pinecone_results.matches[0].metadata.get('template', '')
                    }
                elif llm_matching_template:
                    # Otherwise trust the LLM if it found something
                    final_match = {
                        "id": llm_matching_template.get('ID'),
                        "title": llm_matching_template.get('title'),
                        "method": "Semantic Analysis",
                        "template_content": llm_matching_template.get('template', '')
                    }
                elif has_vector_match:
                    # Fall back to vector similarity
                    final_match = {
                        "id": pinecone_results.matches[0].id,
                        "title": pinecone_results.matches[0].metadata.get('title'),
                        "method": "Content Pattern Match",
                        "template_content": pinecone_results.matches[0].metadata.get('template', '')
                    }
                else:
                    final_match = None

                # Display a simplified, focused result
                st.subheader("Analysis Results")

                # Create three columns for a cleaner display
                col1, col2 = st.columns(2)

                if final_match:
                    # Left column: Show template match details
                    with col1:
                        st.markdown("### Best Matching Template")
                        st.markdown(f"**Title:** {final_match['title']}")
                        st.markdown(f"**ID:** {final_match['id']}")
                        st.markdown(f"**Analysis Method:** {final_match['method']}")

                    # Right column: Show the extracted content
                    with col2:
                        st.markdown("### Analyzed Content")
                        st.text_area("Relevant portion of the alert:", value=relevant_content, height=150, disabled=True)

                    # Display the template content below the columns
                    if final_match.get('template_content'):
                        display_template_content(final_match['template_content'])
                    else:
                        st.warning("Template content not available for display.")
                else:
                    st.error("No matching templates found.")
        else:
            st.warning("Please enter content for analysis.")

# Run the app
if __name__ == "__main__":
    main()
