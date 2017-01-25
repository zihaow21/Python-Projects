from flask import Flask
from flask_ask import Ask, statement, question, session
import json
import requests
import time
import unidecode

app = Flask(__name__) # a typical app definition in flask
ask = Ask(app, "/reddit_reader")  #  ask app, part of the flask app; /: end point, where the ask app would be in the file system

def get_headlines():
    user_pass_dict = {'user': 'zihaow21', 'passwd': 'Wzh131313+', 'api_type': 'json'}

    sess = requests.Session()
    sess.headers.update({'User-Agent': 'I am testing Alexa'})
    sess.post('https://www.reddit.com/api/login', data=user_pass_dict)
    time.sleep(1)
    url = 'https://reddit.com/r/worldnews/.json?limit=10'
    html = sess.get(url)
    data = json.loads(html.content.decode('utf-8'))
    titles = [unidecode.unidecode(listing['data']['title']) for listing in data['data']['children']]

    titles = '... '.join([i for i in titles])

    return titles

#titles = get_headlines()
#print(titles)

@app.route('/') # sets the path of the url, www.emory.edu/
def homepage():
    return "hi there, how you are doing?"

@ask.launch
def start_skill():
    welcome_message = "Hello there, would you like the news?"
    return question(welcome_message)

@ask.intent("YesIntent")
def share_headlines():
    headlines = get_headlines()
    headline_msg = "The current world news headlines are {}".format(headlines)
    return statement(headline_msg)

@ask.intent("NoIntent")
def no_intent():
    bye_text = 'I am no sure why you asked me to run then, but okay ... bye'

if __name__ == '__main__': # run the app
    app.run(debug=True)