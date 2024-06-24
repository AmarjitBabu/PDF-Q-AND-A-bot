import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Path to locally stored model files
model_path = "LaMini-T5-738M"

# Load local tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Function to perform question answering
def answer_question(context, question):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
def main():
    st.title("PDF Question Answering with LaMini-T5-738M")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        full_text = extract_text_from_pdf(uploaded_file)

        st.sidebar.title("Settings")

        # Allow user to select part of the document as context
        st.sidebar.subheader("Select Document Context:")
        doc_parts = full_text.split("\n\n")  # Example: Split by paragraphs

        # Select context from dropdown
        context_choice = st.sidebar.selectbox("Select Context", doc_parts)

        # Provide an area for user-defined context
        context = st.sidebar.text_area("Context", context_choice)

        st.sidebar.subheader("Ask your questions:")
        question = st.sidebar.text_input("Question")

        if st.sidebar.button("Answer"):
            st.subheader("Answer:")
            try:
                st.write(answer_question(context, question))
            except Exception as e:
                st.write("Error:", e)

    st.sidebar.info(
        "This is a demo app for PDF question answering using LaMini-T5-738M."
    )

if __name__ == "__main__":
    main()
