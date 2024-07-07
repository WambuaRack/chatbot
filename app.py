from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a medical-specific model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForCausalLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

app = Flask(__name__)

# Initialize an empty chat history
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids
    msg = request.form["msg"]
    response = get_chat_response(msg)
    print(f"Input: {msg}")
    print(f"Response: {response}")
    return jsonify({"response": response})

def get_chat_response(text):
    global chat_history_ids
    
    # Initialize chat history if not already initialized
    if chat_history_ids is None:
        chat_history_ids = torch.LongTensor([])

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.shape[-1] > 0 else new_user_input_ids

    # generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # decode and return the generated response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update chat history with the current bot response
    chat_history_ids = bot_input_ids
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
