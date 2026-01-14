from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
CHOICE_GENERATOR_MODEL = "gpt-5.2"
CHOICE_GENERATOR_AGENT = None

class Agent:
    def __init__(self, model, temperature=0.7, top_p=1.0, instructions=None):
        self.client = OpenAI()
        self.messages = []
        self.instructions = instructions
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = None
        self.tools = []
        self.max_retries = 3
        self.model = model
        
    def generate_text(self, prompt):
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=self.instructions,
            input=prompt
        )
        return response.output_text

def generate_options(question, answer):
    prompt = f"""
    You are an option generator app. Your task is to provide 3 realistic responses to the question provided.
    Make sure the responses are relevant to the question but are not the correct answer.
    Examples:
    >>>Questin: What is the capital of India?
    >>>Answer: New Delhi
        >>>option:Mumbai/Chennai/Kolkata

        
    Provide a list of 3 options which are not the correct answer in the format: "option1/option2/option3" for the question below.

    Question: {question}
    Answer: {answer}
    """
    global CHOICE_GENERATOR_AGENT, CHOICE_GENERATOR_MODEL
    if CHOICE_GENERATOR_AGENT is None:
        CHOICE_GENERATOR_AGENT = Agent(
            model=CHOICE_GENERATOR_MODEL,
            temperature=0.7,
            instructions="You are an option generator app."
            )
    choices = CHOICE_GENERATOR_AGENT.generate_text(prompt)
    return choices

if __name__ == "__main__":
    print(generate_options("Who is the president of india?", "Droupadi Murmu"))