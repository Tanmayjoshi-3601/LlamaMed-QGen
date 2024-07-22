from transformers import pipeline, BertTokenizer, BertForQuestionAnswering
# from chromadb import ChromaClient
import csv
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

file_path = "knowledge_base.txt"

text = read_text_file(file_path)



# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk
    chunk_overlap=50  # Overlap between chunks
)

# Split the text into chunks
chunks = text_splitter.split_text(text)




# Use a model that is fine-tuned for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

questions = []



response = qa_pipeline({'question': "Generate multiple choice questions from this text:", 'context': chunks[0]})




# Load the T5 model and tokenizer
model_name = "valhalla/t5-base-qg-hl"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load a model for generating distractors
distractor_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Define your text
# text = """Your input text here"""

# # Initialize the RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # Maximum size of each chunk
#     chunk_overlap=50  # Overlap between chunks
# )

# # Split the text into chunks
# chunks = text_splitter.split_text(text)

# Function to generate a question
def generate_question(context):
    input_text = f"generate question: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Function to generate distractors
def generate_distractors(context, answer):
    sentences = context.split('.')
    random.shuffle(sentences)
    distractors = []
    for sentence in sentences:
        if answer.lower() not in sentence.lower():
            distractors.append(sentence.strip())
        if len(distractors) >= 2:
            break
    return distractors if len(distractors) == 2 else []

# Generate questions and options from each chunk
questions = []
# for i, chunk in enumerate(chunks):
question = generate_question(chunks[6])

# Extract the answer from the context (simplified approach)
input_ids = tokenizer.encode(question, return_tensors="pt")
outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

distractors = generate_distractors(chunks[6], answer)

if len(distractors) == 2:
    options = f"Option A: {answer}, Option B: {distractors[0]}, Option C: {distractors[1]}"
    random.shuffle(options.split(", "))  # Randomize the options
    questions.append({'question': question, 'options': options, 'answer': answer})

    # if len(questions) == 50:
    #     break

# Print or process the generated questions
for q in questions:
    print(f"Question: {q['question']}")
    print(f"Options: {q['options']}")
    print(f"Answer: {q['answer']}")
    print()
