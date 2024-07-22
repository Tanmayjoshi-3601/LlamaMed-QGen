import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import csv

# Load Ollama LAMA2 LLM model
llm = Ollama(model="llama2")

# Load the embeddings model and FAISS index
db = FAISS.load_local("faiss_index1", OllamaEmbeddings(), allow_dangerous_deserialization=True)

# Streamlit interface
st.title("MCQ Generator for Anaesthesia Certification Exam")
st.write("Provide a prompt to generate multiple-choice questions (MCQs) for a given topic on anaesthesia.")

prompt_text = st.text_area("Enter the topic or text for generating MCQs:", height=200)
generate_button = st.button("Generate MCQs")

# Function to extract MCQs from the generated text
def extract_mcqs_from_text(text):
    current_mcq = {}
    mcqs = []
    for line in text.split('\n'):
        if line.startswith("Question:"):
            if current_mcq:
                mcqs.append(current_mcq)
                current_mcq = {}
            current_mcq['Question'] = line.replace("Question:", "").strip()
        elif line.startswith("A)") or line.startswith("B)") or line.startswith("C)"):
            option = line[0]
            current_mcq[f'Option_{option}'] = line[2:].strip()
        elif line.startswith("Answer:"):
            current_mcq['Answer'] = line.replace("Answer:", "").strip()
    if current_mcq:
        mcqs.append(current_mcq)
    return mcqs

if generate_button and prompt_text:
    with st.spinner("Generating MCQs..."):
        # Design ChatPrompt Template
        prompt = ChatPromptTemplate.from_template("""
        Generate 3 Multiple choice questions for an anaesthesiology certification exam on the topic of {input}. The topic should be strictly {input} and nothing else. The MCQs generated should have 3 options along with one of the options being the correct answer. Please also give the correct answer at the end. Also don't give any explanations other than the question, options, and answer. It is a strict rule to generate 3 Multiple Choice Questions. Format the output as:
        Question: <question>
        Options:
        A) <option1>
        B) <option2>
        C) <option3>
        Answer: <correct_answer>
        Generate it from this para:  {context}
        """)

        # Create Stuff Document Chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Perform retrieval and generate MCQs
        response = retrieval_chain.invoke({"input": prompt_text})
        text = response['answer']
        mcqs = extract_mcqs_from_text(text)

    # Display MCQs
    if mcqs:
        st.write("Generated MCQs:")
        for mcq in mcqs:
            st.write(f"**Question:** {mcq.get('Question', 'N/A')}")
            st.write(f"A) {mcq.get('Option_A', 'N/A')}")
            st.write(f"B) {mcq.get('Option_B', 'N/A')}")
            st.write(f"C) {mcq.get('Option_C', 'N/A')}")
            st.write(f"**Answer:** {mcq.get('Answer', 'N/A')}")
            st.write("---")
            
            
        # # Saving to CSV
        # with open('mcqs.csv', 'w', newline='') as csvfile:
        #     fieldnames = ['Question', 'Option_A', 'Option_B', 'Option_C', 'Answer']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
        #     writer.writeheader()
        #     for mcq in mcqs:
        #         writer.writerow(mcq)

        # st.write("MCQs have been saved to `mcqs.csv`.")
    else:
        st.write("No MCQs generated. Please try again with a different prompt.")
