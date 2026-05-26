import streamlit as st
import pickle

# Load model
kmeans = pickle.load(open("kmeans.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("Netflix Content Clustering")

user_input = st.text_input("Enter movie description")

if user_input:
    vec = tfidf.transform([user_input])
    cluster = kmeans.predict(vec)

    st.write("Cluster:", cluster)
