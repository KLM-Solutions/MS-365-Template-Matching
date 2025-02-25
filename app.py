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
    page_title="MS365 Template Analysis Tool",
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

# Function to get embedding from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"  # From the image
    )
    return response.data[0].embedding

# Function to analyze content with OpenAI LLM
def analyze_with_llm(content, templates):
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
    {content}
    
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

# Function to upsert a single template to Pinecone
def upsert_to_pinecone(index, data):
    try:
        # Make sure ID is present and valid
        if "ID" not in data:
            return False, "Template is missing ID field"
        
        # Make sure title is present
        if "title" not in data or not data["title"]:
            return False, "Template is missing title field"
        
        # Convert template to vector
        template_text = data.get("template", "") + data.get("title", "")
        vector = get_embedding(template_text)
        
        # Prepare the record for the new Pinecone client
        record = {
            "id": str(data.get("ID")),
            "values": vector,
            "metadata": {
                "name": data.get("name", ""),
                "title": data.get("title", ""),
                "kind": data.get("kind", ""),
                "active": data.get("active", False),
                "template": data.get("template", "")  # Store the full template text in metadata
            }
        }
        
        # Upsert to Pinecone with new client
        index.upsert(vectors=[record])
        return True, "Data successfully upserted to Pinecone!"
    except Exception as e:
        return False, f"Error upserting data: {str(e)}"

# Function to upsert multiple templates to Pinecone
def upsert_templates_to_pinecone(index, templates):
    success_count = 0
    error_count = 0
    errors = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Handling different template formats
    if isinstance(templates, dict) and "templates" in templates:
        # If templates is a dict with a "templates" key
        templates_list = templates["templates"]
    elif isinstance(templates, list):
        # If templates is already a list
        templates_list = templates
    else:
        # Try to convert to list if possible
        try:
            templates_list = list(templates)
        except:
            st.error("Invalid templates format. Please provide a list of template objects.")
            return 0, 1, ["Invalid templates format"]
    
    total_templates = len(templates_list)
    st.info(f"Preparing to upsert {total_templates} templates")
    
    # Process in batches to avoid overwhelming the API
    batch_size = 10
    
    for i in range(0, total_templates, batch_size):
        # Get current batch
        batch = templates_list[i:min(i+batch_size, total_templates)]
        
        for j, template in enumerate(batch):
            try:
                # Update progress
                current_idx = i + j
                progress = (current_idx + 1) / total_templates
                progress_bar.progress(progress)
                status_text.text(f"Processing template {current_idx+1}/{total_templates}...")
                
                # Check template format
                if not isinstance(template, dict):
                    error_count += 1
                    errors.append(f"Template at index {current_idx} is not a valid dictionary")
                    continue
                
                success, message = upsert_to_pinecone(index, template)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Template ID {template.get('ID', 'Unknown')}: {message}")
            except Exception as e:
                error_count += 1
                errors.append(f"Template at index {current_idx}: {str(e)}")
        
        # Small delay between batches to avoid rate limiting
        sleep(1)
    
    # Update the final status
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total_templates} templates")
    
    return success_count, error_count, errors

# Function to parse template file content based on file type
def parse_template_content(content, file_type):
    try:
        if file_type == "json":
            # For JSON files, directly parse the JSON
            parsed_content = json.loads(content)
            
            # Check if it's an array or an object with a templates key
            if isinstance(parsed_content, list):
                return parsed_content
            elif isinstance(parsed_content, dict) and "templates" in parsed_content:
                return parsed_content["templates"]
            else:
                # Try to find any array in the JSON that might be templates
                for key, value in parsed_content.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        return value
                st.warning("JSON structure not recognized. Please make sure it contains an array of templates.")
                return []
        else:
            # For Python/txt files
            # First, try to find a templates list variable
            match = re.search(r'templates\s*=\s*(\[.*?\])', content, re.DOTALL)
            if match:
                template_str = match.group(1)
                # Clean up the string to make it valid JSON
                template_str = re.sub(r'\bNone\b', 'null', template_str)
                template_str = re.sub(r'#.*$', '', template_str, flags=re.MULTILINE)
                template_str = re.sub(r'\bTrue\b', 'true', template_str)
                template_str = re.sub(r'\bFalse\b', 'false', template_str)
                # Fix any remaining Python syntax that's not valid JSON
                template_str = re.sub(r',$\s*\]', ']', template_str, flags=re.MULTILINE)  # Remove trailing commas
                
                try:
                    # Try to parse as JSON first
                    return json.loads(template_str)
                except json.JSONDecodeError:
                    try:
                        # If that fails, try ast.literal_eval
                        return ast.literal_eval(match.group(1))
                    except:
                        st.error("Could not parse the templates list. The format may be invalid.")
                        return []
            else:
                # If no templates list is found, try to find any list of dictionaries
                pattern = r'\[\s*\{\s*"[^"]+"\s*:.*?\}\s*\]'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        template_str = match.group(0)
                        template_str = re.sub(r'\bNone\b', 'null', template_str)
                        template_str = re.sub(r'#.*$', '', template_str, flags=re.MULTILINE)
                        template_str = re.sub(r'\bTrue\b', 'true', template_str)
                        template_str = re.sub(r'\bFalse\b', 'false', template_str)
                        return json.loads(template_str)
                    except:
                        st.error("Found a potential templates list but could not parse it.")
                        return []
                else:
                    # Try to see if there's a single template object in the text
                    try:
                        # Check if this is just a single template (not in a list)
                        if content.strip().startswith('"template"'):
                            # Wrap it in curly braces if it's just a key-value pair
                            content_fixed = "{" + content.strip() + "}"
                            template_obj = json.loads(content_fixed)
                            return [template_obj]
                        elif content.strip().startswith('{') and content.strip().endswith('}'):
                            # It's already a JSON object
                            template_obj = json.loads(content.strip())
                            return [template_obj]
                    except:
                        pass
                        
                    st.error("Could not find a templates list or valid template in the file.")
                    return []
    except Exception as e:
        st.error(f"Error parsing template content: {str(e)}")
        return []

# Function to query Pinecone for similar content
def query_pinecone(index, content):
    try:
        # Convert content to vector
        vector = get_embedding(content)
        
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
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Content Analysis", "Template Management"])
    
    # Tab 1: Content Analysis
    with tab1:
        st.header("Content Analysis")
        st.write("Paste your content below to find the best matching template from Pinecone.")
        
        # Content input with example placeholder
        content = st.text_area("Paste your content here for analysis", height=300,
                              placeholder="In m365-alerts Slack channel\n\nEXAMPLE : \n\nUNUSUAL SIGN IN (FOREIGN_COUNTRY)\nDU = 1810(DON'T PASTE)\n{\n  ..\n  ..(only paste this data) \n  ..\n}\nLink to 365")
        
        if st.button("Analyze Content"):
            if content:
                with st.spinner("Analyzing content..."):
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
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    # Display vector similarity results
                    with col1:
                        st.subheader("Vector Similarity Results")
                        if pinecone_results and hasattr(pinecone_results, 'matches') and pinecone_results.matches:
                            # Get the top match
                            top_match = pinecone_results.matches[0]
                            
                            st.success(f"Top Match: {top_match.metadata.get('title')}")
                            
                            # Display template details
                            st.markdown(f"**ID:** {top_match.id}")
                            st.markdown(f"**Similarity Score:** {top_match.score:.4f}")
                            
                            # Show all matches in an expander
                            with st.expander("View All Similar Templates"):
                                for i, match in enumerate(pinecone_results.matches):
                                    st.markdown(f"**Match {i+1}:** {match.metadata.get('title')} (Score: {match.score:.4f})")
                        else:
                            st.warning("No matching templates found via vector similarity.")
                    
                    # Display LLM analysis results
                    with col2:
                        st.subheader("LLM Analysis Results")
                        if llm_matching_template:
                            st.success(f"Top Match: {llm_matching_template.get('title')}")
                            
                            # Display template details
                            st.markdown(f"**ID:** {llm_matching_template.get('ID')}")
                            st.markdown(f"**Kind:** {llm_matching_template.get('kind')}")
                        else:
                            st.warning("No matching templates found via LLM analysis.")
                    
                    # Show the final recommendation
                    st.subheader("Final Recommendation")
                    
                    # Determine which method to trust more (you can customize this logic)
                    has_vector_match = (pinecone_results and hasattr(pinecone_results, 'matches') and 
                                       len(pinecone_results.matches) > 0)
                    
                    if has_vector_match and pinecone_results.matches[0].score > 0.8:
                        # If vector similarity is very high, trust that
                        final_match = {
                            "id": pinecone_results.matches[0].id,
                            "title": pinecone_results.matches[0].metadata.get('title'),
                            "method": "Vector Similarity (High Confidence)",
                            "template_content": pinecone_results.matches[0].metadata.get('template', '')
                        }
                    elif llm_matching_template:
                        # Otherwise trust the LLM if it found something
                        final_match = {
                            "id": llm_matching_template.get('ID'),
                            "title": llm_matching_template.get('title'),
                            "method": "LLM Analysis",
                            "template_content": llm_matching_template.get('template', '')
                        }
                    elif has_vector_match:
                        # Fall back to vector similarity
                        final_match = {
                            "id": pinecone_results.matches[0].id,
                            "title": pinecone_results.matches[0].metadata.get('title'),
                            "method": "Vector Similarity (Lower Confidence)",
                            "template_content": pinecone_results.matches[0].metadata.get('template', '')
                        }
                    else:
                        final_match = None
                    
                    if final_match:
                        st.success(f"Best Matching Template: {final_match['title']}")
                        st.markdown(f"**Template ID:** {final_match['id']}")
                        st.markdown(f"**Method:** {final_match['method']}")
                        
                        # Display the template content as requested
                        if final_match.get('template_content'):
                            display_template_content(final_match['template_content'])
                        else:
                            st.warning("Template content not available for display.")
                    else:
                        st.error("No matching templates found.")
            else:
                st.warning("Please enter content for analysis.")
    
    # Tab 2: Template Management
    with tab2:
        st.header("Template Management")
        
        # File upload option
        st.subheader("Upload Templates File")
        uploaded_file = st.file_uploader("Upload templates file", type=["py", "txt", "json"])
        
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            # Parse the templates from the file based on its type
            templates = parse_template_content(file_content, file_extension)
            
            if templates and (isinstance(templates, list) or isinstance(templates, dict)):
                # Count templates properly
                if isinstance(templates, list):
                    template_count = len(templates)
                    preview_templates = templates[:min(3, len(templates))]
                else:
                    template_count = 1
                    preview_templates = [templates]
                
                st.success(f"Successfully loaded {template_count} templates.")
                
                # Display template preview safely
                with st.expander("Preview Templates"):
                    st.write(f"First few templates ({min(3, template_count)} of {template_count}):")
                    for i, template in enumerate(preview_templates):
                        if isinstance(template, dict):
                            st.markdown(f"**Template {i+1}**")
                            st.markdown(f"- **Title:** {template.get('title', 'No Title')}")
                            st.markdown(f"- **ID:** {template.get('ID', 'No ID')}")
                            st.markdown(f"- **Name:** {template.get('name', 'No Name')}")
                        else:
                            st.markdown(f"**Template {i+1}**: Invalid format")
                
                if st.button("Upsert All Templates to Pinecone"):
                    with st.spinner("Upserting templates to Pinecone..."):
                        success_count, error_count, errors = upsert_templates_to_pinecone(index, templates)
                        
                        if error_count == 0:
                            st.success(f"Successfully upserted all {success_count} templates!")
                        else:
                            st.warning(f"Upserted {success_count} templates, but encountered {error_count} errors.")
                            with st.expander("View Errors"):
                                for error in errors:
                                    st.error(error)
            else:
                st.error("No templates found in the uploaded file or could not parse the file.")
        
        # Manual template input
        st.subheader("Manual Template Upsert")
        
        with st.expander("Add Single Template"):
            template_data = st.text_area("Enter template data in JSON format", height=200)
            
            if st.button("Upsert Single Template"):
                if template_data:
                    try:
                        # Special handling for single template key-value pair
                        if template_data.strip().startswith('"template"'):
                            template_data = "{" + template_data.strip() + "}"
                            
                        data = json.loads(template_data)
                        with st.spinner("Upserting template..."):
                            success, message = upsert_to_pinecone(index, data)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON format: {str(e)}")
                else:
                    st.warning("Please enter template data before upserting.")

# Run the app
if __name__ == "__main__":
    main()
