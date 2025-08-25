import streamlit as st
import pandas as pd
from time import sleep

# Title of the app
st.title("Count and Listed AI-related Papers from Excel Links")

#  Upload file Excel
uploaded_file = st.file_uploader("Upload file", type="xlsx")
if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    # Use the first column as the link column
    column_name = df.columns[0]
    st.write(f"Operating: {column_name}")
    st.write("Link list:", df.head())

    #  Check if link is valid
    def is_valid_link(link):
        return not (pd.isna(link) or str(link).strip() == "")

    # Function to fetch abstract (simulated)
    def fetch_abstract(link):
        if not is_valid_link(link):
            return "Unvalid or empty link"
        # Simulate fetching abstract
        if "doi" in str(link).lower() or "10." in str(link).lower():  # Check for DOI pattern
            return "This paper discusses artificial intelligence and neural networks."
        return "This paper discusses general topics."

    # Validate and fetch abstracts
    abstracts = []
    for index, row in df.iterrows():
        link = row[column_name]
        abstract = fetch_abstract(link)
        abstracts.append(abstract)
        # Delay to simulate processing time
        sleep(0.1)
        st.write(f"Validate {index + 1}/{len(df)}: {link} -> {abstract}")

    df['Abstract'] = abstracts

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
    st.write("List of articles related to AI:", ai_papers[[column_name, 'Abstract']])

    # Export to Excel
    output = ai_papers.to_excel("AI_articles.xlsx", index=False)
    st.success("Excel file containing the list of AI articles that have been created: AI_articles.xlsx")