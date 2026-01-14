# Knowledge Test App

A Streamlit-based interactive quiz application that tests users' knowledge using questions from the RAG Mini Wikipedia dataset. The app generates multiple-choice options using AI and provides instant feedback on quiz results.

## Features

- **Random Question Selection**: Selects 10 random questions from a curated dataset
- **AI-Generated Options**: Uses OpenAI's GPT model to generate plausible distractors for each question
- **Interactive Quiz Interface**: Clean Streamlit UI with radio buttons for answer selection
- **Session State Management**: Maintains user responses and quiz state across interactions
- **Instant Results**: Displays correct/incorrect answers and final score upon submission
- **Submit Protection**: Submit button is disabled until all questions are answered

## Requirements

- Python 3.8+
- OpenAI API key (for generating answer options)
- Hugging Face token (for accessing datasets)

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will:
1. Load questions from the RAG Mini Wikipedia dataset
2. Generate multiple-choice options for each question
3. Present a 10-question quiz
4. Allow users to select answers
5. Display results after submission

## How It Works

1. **Data Loading**: Questions and answers are loaded from the `rag-datasets/rag-mini-wikipedia` dataset
2. **Option Generation**: For each question, the app uses OpenAI's GPT model to generate 3 incorrect but plausible answer options
3. **Quiz Presentation**: Questions are displayed with shuffled multiple-choice options
4. **Answer Collection**: User selections are stored in Streamlit's session state
5. **Result Calculation**: Upon submission, answers are compared against correct answers to calculate the score

## Dependencies

- `streamlit`: Web app framework
- `datasets`: Hugging Face datasets library
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management
- `torch` & `transformers`: For potential future AI integrations

## Configuration

- Number of questions: Currently set to 10 (configurable in `app.py`)
- AI model: Uses GPT-5.2 for option generation (configurable in `utils.py`)
- Dataset: RAG Mini Wikipedia question-answer pairs

## Contributing

Feel free to submit issues and enhancement requests!