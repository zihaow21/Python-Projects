from flask import Flask
from flask_ask import Ask, statement, question
import json
import requests
import unidecode

app = Flask(__name__) # a typical app definition in flask
ask = Ask(app, "/")  #  ask app, part of the flask app; /: end point, where the ask app would be in the file system


# def get_headlines():
#     user_pass_dict = {'user': 'zihaow21', 'passwd': 'Wzh131313+', 'api_type': 'json'}  # user login info
#
#     sess = requests.Session()  # start a session
#     sess.headers.update({'User-Agent': 'I am testing Alexa'})
#     sess.post('https://www.reddit.com/api/login', data=user_pass_dict)  # pass the credentials
#
#     # api format
#     url = 'https://reddit.com/r/worldnews/.json?limit=10'
#     html = sess.get(url)
#     data = json.loads(html.content.decode('utf-8'))
#
#     # data processing
#     titles = [unidecode.unidecode(listing['data']['title']) for listing in data['data']['children']]
#     titles = '... '.join([i for i in titles])
#
#     return titles


@ask.launch
def launched():
    welcome_message = "Hello there?"
    return question(welcome_message)


@ask.intent("UserInput", default={'sentence': 'hello'})
def share_headlines(sentence):

    # headlines = get_headlines()
    # headline_msg = "The current world news headlines are {}".format(headlines)
    headline_msg = "The message input by user is {}".format(sentence)
    return statement(headline_msg)


if __name__ == '__main__': # run the app
    app.run(debug=True)

### fill out the username and password