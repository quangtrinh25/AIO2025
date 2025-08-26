import streamlit as st
import pandas as pd
import requests # You'll need to install this: pip install requests
from time import sleep
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main App Logic ---

st.set_page_config(layout="wide")
st.title("AI Paper Analyzer 📄�")
st.markdown("Upload an Excel file with a list of DOIs (Digital Object Identifiers) in the first column to fetch their metadata and check for AI relevance.")

# --- Helper Functions ---

def is_valid_link(link):
    """Checks if a link is not null or just empty space."""
    return not (pd.isna(link) or str(link).strip() == "")

def fetch_metadata_from_api(doi):
    """
    Fetches paper metadata first from CrossRef, then tries Semantic Scholar for the abstract if it's missing.
    """
    if not is_valid_link(doi):
        return {"title": "N/A", "journal": "N/A", "year": "N/A", "abstract": "Invalid or empty link"}

    # Clean up the DOI string
    doi = str(doi).replace("https://doi.org/", "").strip()
    
    # --- Step 1: Fetch primary metadata from CrossRef ---
    metadata = {
        "title": "Not Found", "journal": "Not Found", "year": "N/A", "abstract": "Abstract not available"
    }
    
    try:
        crossref_url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(crossref_url, timeout=10)
        response.raise_for_status()
        data = response.json()['message']
        
        metadata['title'] = data.get('title', ['No Title Found'])[0]
        metadata['journal'] = data.get('container-title', ['No Journal Found'])[0]
        
        if 'published-print' in data and 'date-parts' in data['published-print']:
            metadata['year'] = str(data['published-print']['date-parts'][0][0])
        elif 'published-online' in data and 'date-parts' in data['published-online']:
            metadata['year'] = str(data['published-online']['date-parts'][0][0])
        
        # Try to get abstract from CrossRef first
        abstract = data.get('abstract', 'Abstract not available')
        if isinstance(abstract, str) and '<jats:p>' in abstract:
             abstract = abstract.split('<jats:p>')[1].split('</jats:p>')[0]
        metadata['abstract'] = abstract

    except requests.exceptions.RequestException as e:
        logging.warning(f"CrossRef error for DOI {doi}: {e}. Will try Semantic Scholar.")
    except (KeyError, IndexError) as e:
        logging.warning(f"CrossRef parsing error for DOI {doi}: {e}. Will try Semantic Scholar.")

    # --- Step 2: If abstract is missing, try Semantic Scholar ---
    if metadata['abstract'] == 'Abstract not available':
        try:
            s2_url = f'https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=abstract,title,year,journal'
            response = requests.get(s2_url, timeout=10)
            response.raise_for_status()
            s2_data = response.json()

            # Fill in any missing info from CrossRef
            if metadata['title'] == 'Not Found' and s2_data.get('title'):
                metadata['title'] = s2_data['title']
            if metadata['journal'] == 'Not Found' and s2_data.get('journal') and s2_data['journal'].get('name'):
                 metadata['journal'] = s2_data['journal']['name']
            if metadata['year'] == 'N/A' and s2_data.get('year'):
                metadata['year'] = str(s2_data['year'])

            # Get the abstract
            if s2_data.get('abstract'):
                metadata['abstract'] = s2_data['abstract']
            else:
                metadata['abstract'] = 'Abstract not found on Semantic Scholar either.'

        except requests.exceptions.RequestException as e:
            logging.error(f"Semantic Scholar error for DOI {doi}: {e}")
            metadata['abstract'] = "Failed to fetch data from Semantic Scholar."
        except (KeyError, IndexError) as e:
            logging.error(f"Semantic Scholar parsing error for DOI {doi}: {e}")
            metadata['abstract'] = "Could not parse Semantic Scholar response."
            
    return metadata


def analyze_ai_relevance(text):
    """Analyzes text for AI-related keywords."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    keywords = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'computer vision', 'natural language processing']
    return any(keyword in text.lower() for keyword in keywords)

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    if df.empty:
        st.warning("The uploaded file is empty.")
    else:
        # Use the first column as the link column
        doi_column_name = df.columns[0]
        st.write(f"Processing the column: **{doi_column_name}**")
        st.write("First 5 rows of your file:")
        st.dataframe(df.head())

        if st.button("Start Analysis", type="primary"):
            df_to_process = df

            total_rows = len(df_to_process)
            progress_bar = st.progress(0, text="Starting analysis...")
            
            results = []

            for index, row in df_to_process.iterrows():
                doi = row[doi_column_name]
                
                # Update progress text before fetching
                progress_text = f"Processing {index + 1}/{total_rows}: {doi}"
                progress_bar.progress((index + 1) / total_rows, text=progress_text)
                
                # Fetch metadata using the new combined function
                metadata = fetch_metadata_from_api(doi)
                
                # Analyze for AI relevance (checking both title and abstract)
                is_ai_related = analyze_ai_relevance(metadata['title']) or analyze_ai_relevance(metadata['abstract'])
                
                # Store results
                result_row = {
                    doi_column_name: doi,
                    'Title': metadata['title'],
                    'Journal': metadata['journal'],
                    'Year': metadata['year'],
                    'Abstract': metadata['abstract'],
                    'Is AI-Related': is_ai_related
                }
                results.append(result_row)
                
                # Short delay to be polite to the APIs
                sleep(0.5) # Increased delay slightly for the second API

            progress_bar.empty()
            
            # Create a new DataFrame from the results
            results_df = pd.DataFrame(results)
            
            # Filter for AI-related papers
            ai_papers_df = results_df[results_df['Is AI-Related']].copy()

            st.success(f"**Analysis Complete!** Found **{len(ai_papers_df)}** AI-related papers.")

            if not ai_papers_df.empty:
                st.write("### AI-Related Papers:")
                st.dataframe(ai_papers_df)

                # Convert to Excel format in memory
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    ai_papers_df.to_excel(writer, index=False, sheet_name='AI_Papers')
                
                st.download_button(
                    label="📥 Download AI Papers as Excel",
                    data=output.getvalue(),
                    file_name="AI_articles_found.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No papers matching the AI keywords were found.")

            st.write("### All Processed Papers:")
            st.dataframe(results_df)
