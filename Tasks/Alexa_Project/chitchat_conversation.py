from flask import Flask
from flask_ask import Ask, statement, question
from Tasks.Chatbot_Practice.chitchat_component import ChitChat


app = Flask(__name__)
ask = Ask(app, "/chitchat")

def chitchat(query):
    cc = ChitChat()
    response = cc.mainTest(query)
    return response

@app.route('/')
def homepage():
    return "Hi, there how are you?"

@ask.launch
def start_skill():
    welcome_message = "Hello there, what's up?"
    return question(welcome_message)

@ask.intent("ChatIntent")
def conversation(query):
    response = chitchat(query)
    return statement(response)

@ask.intent("ExitIntent")
def exit_intent():
    exit_message = "It was greating talking to you. Have a great one!"
    return statement(exit_message)

if __name__ == '__main__':
    app.run(debug=True)