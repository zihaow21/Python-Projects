from flask import Flask, render_template
from flask_ask import Ask, statement, question, session, request, context
import logging
from Tasks.Chatbot_Practice.chitchat_component import ChitChat
from news_retrieve import NewsRetrival
from weather_forcast import WeatherForecast
import json


newsRetrieval = NewsRetrival()
# tokenizer = RegexpTokenizer(r'\w+')

app = Flask(__name__)
ask = Ask(app, "/emersonbot")
log = logging.getLogger('flask_ask').setLevel(logging.DEBUG)

# def chitchat(query):
#     cc = ChitChat()
#     response = cc.mainTest(query)
#     return response


@app.route('/')
def homepage():
    return "Hi, there how are you?"

@ask.launch
def start_skill():
    welcome_message = "Hello there, my name is emerson bot, what can I do for you?"
    return question(welcome_message).reprompt("hello are you there?")

@ask.on_session_started
def new_session():
    log.info('new session started')

@ask.intent("NewsInput")
def newsComponent(topics, usplaces, regions, cities):
    topics = topics
    usplaces = usplaces
    regions = regions
    cities = cities
    session.attributes = dict()
    if topics:
        data = newsRetrieval.search("{}, {}, {}, {}".format(topics, usplaces, regions, cities))

        if len(data) == 2:
            result = data[1]
            result = ''.join([i if ord(i) < 128 else ' ' for i in result])
            return question(result + ". sorry, I know this is a bad joke. So, what can I do for you now?")

        else:
            headlines = data['headline']
            print headlines
            headlines = ''.join([i if ord(i) < 128 else ' ' for i in headlines])
            body = data['body']
            body = ''.join([i if ord(i) < 128 else ' ' for i in body])
            session.attributes["body"] = body

            return question("here is the news headlines. {}. Would you like the details of the news?".format(headlines))

# @ask.intent("AMAZON.YesIntent")
# def news_detail():
#     body = session.attributes["body"]
#     return question("here is the news details. {}".format(body)).reprompt("what is your opinion?")

@ask.intent("WeatherInfo")
def weather_today(loc):
    session.attributes = dict()
    if loc == None:
        location = "atlanta"
    else:
        location = loc

    print "the session format is {}".format(session)

    forcast = WeatherForecast(location)
    current_info, forecasts_info = forcast.weatherInfo()
    session.attributes["forecasts"] = forecasts_info
    return question(current_info + ". would you like forecasts for the following five days?")

@ask.intent("AMAZON.YesIntent")
def yes_specific_match():
    key = session.attributes.keys()[0]
    if key == "forecasts":
        forecasts_info = session.attributes["forecasts"]
        return question(forecasts_info + ". so, what is your plan today?")
    elif key == "body":
        body = session.attributes["body"]
        return question("here is the news details. {}. Would you like to have some comments".format(body))

@ask.intent("GeneralNoIntent")
def no_specific_match(response):
    key = session.attributes.keys()[0]
    if key == "forecasts":
        return question("OK, then. what would you like now. Some jokes?")
    elif key == "body":
        return question("OK, then. what would you like now. Some jokes?")

# @ask.intent("Amazon.NoInTent")
# def no_specific_match(response):
#     key = session.attributes.keys()[0]
#     if key == "forecasts":
#         return question("OK, then. what can I do for you now")
#     elif key == "body":
#         return question("OK, then. what can I do for you now")

@ask.intent("GeneralUtterance")
def general(sentence):
    print "I have not been prepared well yet, so I am echoing you. {}".format(sentence)
    return statement("{}".format(sentence))

@ask.intent("AMAZON.StopIntent")
def exit_intent():
    exit_message = "It was greating talking to you. Have a great one!"
    return statement(exit_message)

@ask.session_ended
def session_ended():
    return "", 500

if __name__ == '__main__':
    app.run(debug=True)
