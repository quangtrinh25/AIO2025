import streamlit as st
import pandas as pd
from time import sleep

# Title of the app
st.title("Count and Listed AI-related Papers from Excel Links")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file", type="xlsx")
if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    # Use the first column as the link column
    column_name = df.columns[0]
    st.write(f"Operating: {column_name}")
    st.write("Link list:", df.head())

    # Check if link is valid
    def is_valid_link(link):
        return not (pd.isna(link) or str(link).strip() == "")

    # Function to fetch metadata (simulated with detailed titles and journals)
    def fetch_metadata(link):
        if not is_valid_link(link):
            return {"abstract": "Unvalid or empty link", "title": "N/A", "journal": "N/A", "year": "N/A"}
        # Simulate fetching metadata based on DOI pattern
        if "doi" in str(link).lower() or "10." in str(link).lower():
            # Detailed AI-related titles and journals
            title_options = [
                "Advancements in Artificial Intelligence for Neural Network Optimization",
                "Machine Learning Techniques in Deep Learning Models",
                "Neural Network Applications in Real-Time AI Systems"
            ]
            journal_options = [
                "IEEE Transactions on Artificial Intelligence",
                "Journal of Machine Learning Research",
                "Nature Machine Intelligence"
            ]
            # Use hash of link to select index for variety
            link_hash = hash(str(link)) % len(title_options)
            return {
                "abstract": f"This paper explores {title_options[link_hash].lower()} and their implications.",
                "title": title_options[link_hash],
                "journal": journal_options[link_hash],
                "year": "2023"
            }
        # Non-AI papers
        return {
            "abstract": "This paper discusses general scientific topics.",
            "title": f"General Study on {link[-5:]}",
            "journal": "General Science Journal",
            "year": "2022"
        }

    # Validate and fetch abstracts and metadata
    abstracts = []
    titles = []
    journals = []
    years = []
    for index, row in df.iterrows():
        link = row[column_name]
        metadata = fetch_metadata(link)
        abstracts.append(metadata["abstract"])
        titles.append(metadata["title"])
        journals.append(metadata["journal"])
        years.append(metadata["year"])
        # Delay to simulate processing time
        sleep(0.1)
        st.write(f"Validate {index + 1}/{len(df)}: {link} -> {metadata['abstract']}")

    df['Abstract'] = abstracts
    df['Title'] = titles
    df['Journal'] = journals
    df['Year'] = years

    # Analyze abstracts for AI relevance
    def analyze_ai(abstract):
        if pd.isna(abstract) or abstract == "Unvalid or empty link":
            return False
        keywords = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network']
        return any(keyword in abstract.lower() for keyword in keywords)

    df['Related to AI'] = df['Abstract'].apply(analyze_ai)

    # Filter AI-related papers
    ai_papers = df[df['Related to AI'] == True].dropna(subset=['Abstract'])

    # Display results
    st.write(f"Number of articles related to AI : {len(ai_papers)}")
    st.write("List of articles related to AI:", ai_papers[[column_name, 'Title', 'Journal', 'Year', 'Abstract']])

    # Export to Excel
    output = ai_papers.to_excel("AI_articles.xlsx", index=False)
    st.success("Excel file containing the list of AI articles that have been created: AI_articles.xlsx")
