import openai
import os

# Load the API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.6
    )

    output_text = response.choices[0].message['content']
    return output_text

def main():
    print("ChatBot\nAsk me anything!")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = get_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
