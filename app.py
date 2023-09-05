import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings

# Check if environment variables are present. If not, throw an error
if st.secrets["OPENAPI_KEY"] is None:
    st.error("OPENAPI_KEY not set. Please set this environment variable and restart the app.")

openai.api_key = st.secrets["OPENAPI_KEY"]
tab1, tab2 = st.tabs(["Search", "Recommendation"])

def get_embeds(query):
    return openai.Embedding.create(input = query, engine='text-embedding-ada-002')['data'][0]['embedding']   

with tab1:
    st.title("Podcast Search 🔎")
    query = st.text_input("Type somethings to search podcast")

    if st.button("Search"):
        df = pd.read_csv('processed.csv')
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        
        query_embeds = get_embeds(query) 
        df['distances'] = distances_from_embeddings(query_embedding = query_embeds, embeddings = df['embeddings'])

        sorted = df.sort_values('distances', ascending=True).head(10)
        st.dataframe(sorted, hide_index=True, column_order=['title', 'url', 'distances'])


with tab2:
    df = pd.read_csv('processed.csv')
    col1, col2 = st.columns(2)

    with col1:
        "The list of podcasts in the database"
        st.dataframe(df, column_order=['title', 'standFirst'])

    with col2:
        query_for_rec = st.text_input("Type podcast title (for better result, title + standfirst + body) to see recommendattions")

        if st.button("Find Recommendation"):
            df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        
            query_embeds = get_embeds(query_for_rec) 
            df['distances'] = distances_from_embeddings(query_embedding = query_embeds, embeddings = df['embeddings'])

            sorted = df.sort_values('distances', ascending=True).head(10)
            st.dataframe(sorted, hide_index=True, column_order=['title', 'url', 'distances'])
