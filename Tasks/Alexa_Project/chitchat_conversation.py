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
def newsComponent(news, time, usplaces, regions, cities):

    news = news
    time = time
    usplaces = usplaces
    regions = regions
    cities = cities
    if news:
        data = newsRetrieval.search("{}, {}, {}, {}".format(news, time, usplaces, regions, cities))

        if len(data) == 2:
            result = data[1]
            result = ''.join([i if ord(i) < 128 else ' ' for i in result])
            return question(result).reprompt("sorry, I know this is a bad joke. So, what can I do for you now?")

        else:
            headlines = data['headline']
            headlines = ''.join([i if ord(i) < 128 else ' ' for i in headlines])
            body = data['body']
            body = ''.join([i if ord(i) < 128 else ' ' for i in body])
            session.attributes = dict()
            session.attributes["body"] = body
            return question("here is the news headlines. {}".format(headlines)).reprompt("Would you like the details of the news?")

# @ask.intent("AMAZON.YesIntent")
# def news_detail():
#     body = session.attributes["body"]
#     return question("here is the news details. {}".format(body)).reprompt("what is your opinion?")

@ask.intent("WeatherInfo")
def weather_today(loc):
    if loc == None:
        location = "atlanta"
    else:
        location = loc

    print "the session format is {}".format(session)

    forcast = WeatherForecast(location)
    session.attributes = dict()
    current_info, forecasts_info = forcast.weatherInfo()
    session.attributes["forecasts"] = forecasts_info
    return question(current_info).reprompt("would you like forecasts for following five days?")

@ask.intent("AMAZON.YesIntent")
def specific_match():
    key = session.attributes.keys()[0]
    if key == "forecasts":
        forecasts_info = session.attributes["forecasts"]
        return question(forecasts_info).reprompt("what is your plan today?")
    elif key == "body":
        body = session.attributes["body"]
        return question("here is the news details. {}".format(body)).reprompt("what is your opinion?")

# @ask.intent("AMAZON.YesIntent")
# def weather_forecasts():
#     forecasts_info = session.attributes["forecasts"]
#     return question(forecasts_info).reprompt("what is your plan today?")

@ask.intent("GeneralUtterance")
def general(sentence):
    print "I have not been prepared well yet, so I am echoing you. {}".format(sentence)
    return statement("{}".format(sentence))


@ask.intent("AMAZON.StopIntent")
def exit_intent():
    exit_message = "It was greating talking to you. Have a great one!"
    return statement(exit_message)

if __name__ == '__main__':
    app.run(debug=True)
