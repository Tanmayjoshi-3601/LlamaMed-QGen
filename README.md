# LlamaMed-QGen

## Overview

LlamaMed-QGen is a project designed to generate multiple-choice questions (MCQs) for anaesthesia certification exams. The project leverages LangChain, Ollama, and Streamlit to create an efficient and user-friendly application for generating MCQs based on text and images.

## Features

- **Text and Image-based MCQ Generation**: Generate high-quality MCQs from both text and image inputs.
- **LangChain Integration**: Utilize LangChain for prompt engineering and efficient processing.
- **Ollama Integration**: Incorporate Ollama for text generation.
- **Streamlit Interface**: Provide a simple and interactive web interface for users to generate and view MCQs.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/Tanmayjoshi-3601/LlamaMed-QGen.git
cd LlamaMed-QGen
```

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage
### Running the Application

- Navigate to the Project Directory: Open your terminal and navigate to the project directory.

```bash
cd LlamaMed-QGen
```

- Run the Streamlit App:
```bash
streamlit run app.py
```
- Open Your Browser: Go to http://localhost:8501 in your web browser to access the application.


## Generating MCQs

  - Upload Text or Image: Use the interface to upload a text file or an image.
  - Generate MCQs: Click on the "Generate MCQs" button to create multiple-choice questions based on the uploaded content.
  - View Results: The generated MCQs will be displayed on the same page.

## Project Structure
  
  - app.py: The main Streamlit application file.
  - requirements.txt: List of dependencies required to run the project.
    
