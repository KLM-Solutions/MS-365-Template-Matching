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

# List of template descriptions
template_descriptions = [
    {"id": 565, "title": "Record Type 28 - ThreatIntelligence", "description": "Alerts for phishing or malware events detected by Exchange Online Protection or Microsoft Defender for Office 365."},
    {"id": 568, "title": "Record Type 15 - AzureActiveDirectorySTSLogin", "description": "Secure Token Service (STS) logon events in Azure Active Directory."},
    {"id": 569, "title": "Login Failures Using Legacy Authentication", "description": "Failed login attempts using outdated authentication methods (BAV2ROPC, Basic Authentication)."},
    {"id": 570, "title": "Record Type 8 - Add/Delete Users", "description": "Administrative events related to user account creation or deletion in Azure Active Directory."},
    {"id": 572, "title": "Login Failures for Disabled Accounts", "description": "Login attempts on accounts that have been disabled in Active Directory (Error Code 50057)."},
    {"id": 573, "title": "Configuring anti-malware policies in EOP", "description": "Information on setting up and managing anti-malware policies in Exchange Online Protection."},
    {"id": 577, "title": "How to Fight Phishing in O365 Tenants", "description": "Guide for implementing anti-phishing measures in Microsoft 365, including email authentication and security awareness."},
    {"id": 578, "title": "Enabling MFA on your 365 Tenant", "description": "Instructions for implementing multi-factor authentication in Microsoft 365 environments."},
    {"id": 590, "title": "Alert Detected for Office 365", "description": "Generic template for Security and Compliance center alerts."},
    {"id": 591, "title": "Users targeted by Phishing/Malware campaigns", "description": "Identification of accounts that are most frequently targeted by phishing attempts."},
    {"id": 626, "title": "Suspicious login activity with impossible travel detected", "description": "Alerts for login activity from geographically distant locations within a short time frame."},
    {"id": 632, "title": "Unusual amount of login failures", "description": "Detection of higher than normal failed login attempts, possible brute force attack indicator."},
    {"id": 642, "title": "Gradient 365 alert: Impossible travel Sign-in detected", "description": "Alert for login activity from different countries within an improbable timeframe."},
    {"id": 646, "title": "Suspicious Account Manipulation", "description": "Detection of changes made to user accounts, possibly for maintaining unauthorized access."},
    {"id": 651, "title": "Gradient 365 alert: Unusual sign-in activity detected", "description": "Logins from previously unseen IP addresses or locations."},
    {"id": 652, "title": "Gradient 365 alert: Sign-in from a Blacklisted IP detected", "description": "Logins from IP addresses known to be associated with malicious activity."},
    {"id": 653, "title": "Gradient 365 alert: Sign-in from an Anonymous IP detected", "description": "Logins through anonymizing services like Tor."},
    {"id": 658, "title": "MS365 - User Accounts Added or Deleted", "description": "Monitoring of user account creation and deletion events."},
    {"id": 659, "title": "Gradient 365 alert: Users targeted by Phishing/Malware campaigns", "description": "Detailed alert about accounts frequently targeted by phishing attempts."},
    {"id": 660, "title": "Protecting Microsoft 365 from on-premises attacks", "description": "Best practices for securing Microsoft 365 from attacks originating on-premises."},
    {"id": 664, "title": "MS365 - Initial Sign-in Report", "description": "Baseline report of user sign-ins over a 7-day period to establish normal patterns."},
    {"id": 668, "title": "MS365 - Unusual Sign-in Report", "description": "Summary of sign-ins that don't match established baselines but aren't necessarily malicious."},
    {"id": 683, "title": "User clicked through potentially malicious URL", "description": "Alert when a user bypasses Safe Links warnings to access a suspicious URL."},
    {"id": 690, "title": "Activity from infrequent country", "description": "Detection of sign-ins from countries not normally associated with organization activity."},
    {"id": 691, "title": "Suspicious Inbox Manipulation", "description": "Alert for suspicious email rules that might hide or redirect incoming messages."},
    {"id": 697, "title": "New Client Onboarding", "description": "Welcome message and security documentation for new clients."},
    {"id": 700, "title": "MS365 - New Client Onboarding", "description": "Structured onboarding information for new Microsoft 365 clients."},
    {"id": 701, "title": "MFA disabled", "description": "Alert when multi-factor authentication is disabled for an account."},
    {"id": 702, "title": "Gradient 365 alert: Suspicious Password Change", "description": "Alert for potentially unauthorized password changes."},
    {"id": 704, "title": "Elevation of Exchange admin privilege", "description": "Detection when Exchange admin privileges are granted to a user."},
    {"id": 708, "title": "Microsoft Defender Alert - Anonymous Login", "description": "Alert for sign-ins through anonymizing services."},
    {"id": 709, "title": "Gradient 365 alert: Recurring Login Failures", "description": "Detection of repeated login failures for specific accounts over time."},
    {"id": 715, "title": "Suspicious Inbox Rule Creation", "description": "Alert when inbox rules are created that could be used for data exfiltration."},
    {"id": 716, "title": "Test Sign-in from an Anonymous IP detected Test", "description": "Test template for anonymous IP login detection."},
    {"id": 727, "title": "MS365 - Device no longer compliant", "description": "Alert when a device falls out of compliance with security policies."},
    {"id": 728, "title": "Anomalous Microsoft 365 Traffic", "description": "Detection of unusual activity patterns in Microsoft 365."},
    {"id": 734, "title": "MS365 - Suspicious File Deletion activity", "description": "Alert for unusual file deletion events in SharePoint or OneDrive."},
    {"id": 735, "title": "MS365 - Suspicious Folder Deletion", "description": "Alert for unusual folder deletion events in SharePoint or OneDrive."},
    {"id": 741, "title": "Atypical travel", "description": "Microsoft Defender alert for suspicious sign-ins from disparate locations."},
    {"id": 742, "title": "Connection to a custom network indicator", "description": "Alert when connections are made to custom-defined suspicious network locations."},
    {"id": 743, "title": "Malicious IP address", "description": "Detection of connections from known malicious IP addresses."},
    {"id": 744, "title": "High Severity Alert Detected", "description": "Generic template for high severity Microsoft Defender alerts."},
    {"id": 745, "title": "Activity from a password-spray associated IP address", "description": "Detection of login attempts from IPs associated with password spray attacks."},
    {"id": 746, "title": "Unwanted software was detected during a scheduled scan", "description": "Alert for potentially unwanted applications found during security scans."},
    {"id": 747, "title": "DLP-Content matches U.S. Health Insurance Act (HIPAA)", "description": "Alert when Data Loss Prevention detects potential HIPAA-protected information."},
    {"id": 748, "title": "Malware was prevented", "description": "Notification that malware was blocked before execution."},
    {"id": 749, "title": "Suspicious email sending patterns detected", "description": "Early warning for unusual email sending behavior that might indicate account compromise."},
    {"id": 750, "title": "User restricted from sending email", "description": "Alert when a user is blocked from sending email due to suspicious activity."},
    {"id": 751, "title": "Malware was blocked", "description": "Alert for successful blocking of malware."},
    {"id": 764, "title": "Suspicious connection blocked by network protection", "description": "Notification of blocked connections to malicious websites."},
    {"id": 786, "title": "Gradient 365 alert: Foreign country activity detected", "description": "Alert for login activity from unexpected foreign countries."},
    {"id": 787, "title": "Gradient 365 alert: Suspicious Password Change Cont'd", "description": "Follow-up alert for suspicious password change patterns."},
    {"id": 789, "title": "Gradient 365 alert: Successful Phish Delivered to Inbox", "description": "Alert when a phishing email successfully bypasses filters."},
    {"id": 807, "title": "Gradient 365 alert: A potentially Malicious URL click was detected", "description": "Detection of user clicks on suspicious URLs."},
    {"id": 808, "title": "Gradient 365 alert: Multiple failed user logon attempts to a service", "description": "Alert for repeated authentication failures to specific services."},
    {"id": 809, "title": "Gradient 365 alert: Connection to adversary-in-the-middle (AiTM) phishing site", "description": "Detection of connections to known phishing sites using man-in-the-middle tactics."},
    {"id": 810, "title": "Gradient 365 alert: Leaked Credentials", "description": "Alert when user credentials are found in known data breaches."},
    {"id": 811, "title": "Gradient 365 alert: Email sending limit exceeded", "description": "Notification when a user exceeds allowed email sending thresholds."},
    {"id": 834, "title": "Gradient 365 alert: Unusual sign-in failure detected (Foreign Country)", "description": "Alert for failed login attempts from unexpected foreign locations."}
]

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

# Function to extract alert type from content
def extract_alert_type(content):
    """
    Extract the alert type from the content - usually found at the beginning
    """
    try:
        # Look for the first few lines before DU number
        lines = content.split('\n')
        alert_type = []
        
        for line in lines:
            if line.strip() and 'DU =' not in line and not line.strip().startswith('{'):
                alert_type.append(line.strip())
            if 'DU =' in line:
                break
                
        return ' '.join(alert_type).strip()
    except Exception as e:
        st.error(f"Error extracting alert type: {str(e)}")
        return ""

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

# Function to identify template based on the alert type and description
def identify_template(alert_type, content):
    """
    First identify the template based on alert type and content before going to Pinecone
    """
    # Create a prompt that emphasizes using just the list of templates
    system_prompt = """
    You are an AI assistant specialized in identifying MS365 security alerts based on their type and content.
    Your task is to match the provided alert to the most appropriate template from a predefined list.
    Analyze the alert type and content carefully and determine which template would be most suitable.
    Return ONLY the ID of the best matching template as a number without any additional text.
    """
    
    user_prompt = f"""
    Alert type: {alert_type}
    
    Alert content:
    {content}
    
    Available templates:
    {json.dumps(template_descriptions, indent=2)}
    
    Based on the alert type and content, which template ID is the best match? Respond with only the template ID as a number.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10
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
        st.error(f"Error identifying template: {str(e)}")
        return None

# Function to analyze content with OpenAI LLM (backup method)
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10
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

# Function to get template by ID from Pinecone
def get_template_by_id(index, template_id):
    try:
        # First try to fetch directly by ID if the ID is a string (assuming IDs are used as vector IDs in Pinecone)
        try:
            result = index.fetch(ids=[str(template_id)])
            if result and str(template_id) in result.vectors:
                vector_data = result.vectors[str(template_id)]
                return {
                    "ID": template_id,
                    "title": vector_data.metadata.get("title", ""),
                    "name": vector_data.metadata.get("name", ""),
                    "kind": vector_data.metadata.get("kind", ""),
                    "active": vector_data.metadata.get("active", False),
                    "template": vector_data.metadata.get("template", "")
                }
        except Exception as fetch_error:
            st.warning(f"Could not fetch template directly by ID: {str(fetch_error)}")
        
        # If direct fetch fails, use a query with a dummy vector to find all and filter
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True
        )
        
        for match in results.matches:
            # Try to match by ID
            if match.id == str(template_id) or int(match.id) == int(template_id):
                return {
                    "ID": int(match.id) if match.id.isdigit() else match.id,
                    "title": match.metadata.get("title", ""),
                    "name": match.metadata.get("name", ""),
                    "kind": match.metadata.get("kind", ""),
                    "active": match.metadata.get("active", False),
                    "template": match.metadata.get("template", "")
                }
                
        return None
    except Exception as e:
        st.error(f"Error fetching template by ID: {str(e)}")
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

# Function to find template from predefined list
def find_template_in_list(template_id):
    for template in template_descriptions:
        if template["id"] == template_id:
            return template
    return None

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
    st.title("MS-365 Template_Matching")
    
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
                # Extract the alert type and relevant portion
                alert_type = extract_alert_type(content)
                relevant_content = extract_relevant_content(content)
                
                # FIRST APPROACH: Directly identify the template based on alert type and content
                template_id = identify_template(alert_type, relevant_content)
                
                if template_id:
                    st.subheader(f"Initial Template Match: ID {template_id}")
                    
                    # Find template in predefined list
                    template_info = find_template_in_list(template_id)
                    
                    if template_info:
                        st.write(f"**Title:** {template_info['title']}")
                        st.write(f"**Description:** {template_info['description']}")
                    
                    # Now fetch the full template content from Pinecone
                    with st.spinner("Fetching template details from database..."):
                        template_from_db = get_template_by_id(index, template_id)
                        
                        if template_from_db and template_from_db.get("template"):
                            st.success("Template found in database!")
                            
                            # Create two columns for a cleaner display
                            col1, col2 = st.columns(2)
                            
                            # Left column: Show template match details
                            with col1:
                                st.markdown("### Template Details")
                                st.markdown(f"**Title:** {template_from_db.get('title', '')}")
                                st.markdown(f"**ID:** {template_from_db.get('ID', '')}")
                                st.markdown(f"**Name:** {template_from_db.get('name', '')}")
                                st.markdown(f"**Status:** {'Active' if template_from_db.get('active') else 'Inactive'}")
                            
                            # Right column: Show the extracted content
                            with col2:
                                st.markdown("### Analyzed Content")
                                st.text_area("Alert type:", value=alert_type, height=50, disabled=True)
                                st.text_area("Relevant portion of the alert:", value=relevant_content, height=150, disabled=True)
                            
                            # Display the template content below the columns
                            display_template_content(template_from_db.get('template', ''))
                        else:
                            st.warning(f"Template ID {template_id} found but no content available in database.")
                            
                            # FALLBACK APPROACH: Vector search with Pinecone
                            st.info("Falling back to content-based search...")
                            pinecone_results = query_pinecone(index, content)
                            
                            if pinecone_results and hasattr(pinecone_results, 'matches') and len(pinecone_results.matches) > 0:
                                fallback_match = {
                                    "id": pinecone_results.matches[0].id,
                                    "title": pinecone_results.matches[0].metadata.get('title'),
                                    "method": "Content Pattern Match",
                                    "template_content": pinecone_results.matches[0].metadata.get('template', '')
                                }
                                
                                st.markdown("### Best Matching Template from Content Analysis")
                                st.markdown(f"**Title:** {fallback_match['title']}")
                                st.markdown(f"**ID:** {fallback_match['id']}")
                                
                                if fallback_match.get('template_content'):
                                    display_template_content(fallback_match['template_content'])
                                else:
                                    st.warning("Template content not available for display.")
                            else:
                                st.error("No matching templates found via content search.")
                else:
                    st.error("Could not identify a template for this alert. Trying alternative methods...")
                    
                    # FALLBACK APPROACHES
                    # 1. Try Pinecone vector search
                    pinecone_results = query_pinecone(index, content)
                    
                    # 2. LLM analysis with all templates (if vector search fails)
                    all_templates = get_all_templates_from_pinecone(index)
                    llm_template_id = analyze_with_llm(content, all_templates) if all_templates else None
                    
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
                        st.success("Found a match using content pattern analysis.")
                    elif llm_template_id:
                        # Find the template with the matching ID from LLM analysis
                        llm_matching_template = None
                        for template in all_templates:
                            if str(template.get("ID")) == str(llm_template_id):
                                llm_matching_template = template
                                break
                                
                        if llm_matching_template:
                            final_match = {
                                "id": llm_matching_template.get('ID'),
                                "title": llm_matching_template.get('title'),
                                "method": "Semantic Analysis",
                                "template_content": llm_matching_template.get('template', '')
                            }
                            st.success("Found a match using semantic analysis.")
                        else:
                            final_match = None
                    elif has_vector_match:
                        # Fall back to vector similarity
                        final_match = {
                            "id": pinecone_results.matches[0].id,
                            "title": pinecone_results.matches[0].metadata.get('title'),
                            "method": "Content Pattern Match",
                            "template_content": pinecone_results.matches[0].metadata.get('template', '')
                        }
                        st.success("Found a match using content pattern analysis.")
                    else:
                        final_match = None
                    
                    # Display fallback results
                    if final_match:
                        # Create two columns for a cleaner display
                        col1, col2 = st.columns(2)
                        
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
                        st.error("No matching templates found through any method.")
        else:
            st.warning("Please enter content for analysis.")

# Run the app
if __name__ == "__main__":
    main()
