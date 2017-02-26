from flask import Flask
from flask_ask import Ask, statement, question
from Context import RequestContext

app = Flask(__name__) # a typical app definition in flask
ask = Ask(app, "/")  #  ask app, part of the flask app; /: end point, where the ask app would be in the file system


@ask.launch
def launched():
    welcome_message = "Hello there, what's up?"
    return question(welcome_message)


@ask.intent("UserInput", default={'sentence': 'hello'})
def current_request(sentence):
    headline_msg = "The message input by user is {}".format(sentence)
    context = RequestContext()
    return statement(headline_msg)


if __name__ == '__main__': # run the app
    app.run(debug=True)
