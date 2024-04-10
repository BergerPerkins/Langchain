import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user request only based on the given context"),
        ("user", "Question:{question}\nContext:{context}")
    ]
)

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(question, context):
    input_text = prompt.format_prompt(question=question, context=context).to_string()
    input_ids = torch.tensor(tokenizer.encode(input_text, return_tensors="pt")).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

question = "Can you summarize the speech?"
context = """
Even though you are only a very small speck of ocean, of the 300 million children on whose shoulders the future of India rests, you can ignite the minds, light the fire, become a burning candle to light another candle. My dear children, you work in the school for about 25000 hours before you complete 12th std. Within the school curriculum / school hours, I want you to contribute to the Mission of Developed India by involving in the student centric activities like Literacy Development, Eco-Care Movement.
India after its independence was determined to move ahead with planned policies for Science & Technology. Now, India is very near to self-sufficiency in food, making the ship to mouth existence of 1950s, an event of the past. Also improvements in the health sector, have eliminated few contagious diseases. There is a increase in life expectancy. Small scale industries provide high percentage of National GDP - a vast change in 1990s compared to 1950s. Today India can design, develop and launch world class geo-stationary and sun synchronous, remote sensing satellites."""
print(generate_response(question, context))
