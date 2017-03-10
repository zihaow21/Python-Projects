from flask import Flask, render_template
from flask_ask import Ask, statement, question, session, request
import logging
from news_retrieve import NewsRetrival
from weather_forcast import WeatherForecast
from AIML_Bot.make_chatterbot import ChitChat
from termcolor import colored
import config


newsRetrieval = NewsRetrival()
chitchat = ChitChat()

app = Flask(__name__)
ask = Ask(app, "/emersonbot")
log = logging.getLogger('flask_ask').setLevel(logging.DEBUG)

def global_count():
    config.COUNT += 1

def global_count_reset():
    config.Count = 0

def global_param_reset():
    config.CURRENT_PARAMS = []

def global_content_reset():
    config.CURRENT_CONTENT = None

def global_all_reset():
    config.CURRENT_SESSION = ""
    config.PREVIOUS_SESSION = ""
    config.COUNT = 0
    config.HISTORY = []
    config.CURRENT_CONTENT = None
    config.CURRENT_PARAMS = []
    config.PREVIOUS_PARAMS = []
    config.NEXT_INTENTION = ""

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


def search_news(topics, dates, usplaces, regions, cities):
    data = newsRetrieval.search("{}, {}, {}, {}, {}".format(topics, dates, usplaces, regions, cities))
    return data

def news_response_construct():
    history_temp = []
    history_temp.append(config.CURRENT_PARAMS)
    data = config.CURRENT_CONTENT[config.COUNT]["_source"]

    if len(data) == 2:
        result = data[1]
        result = ''.join([i if ord(i) < 128 else ' ' for i in result])
        response = result + ". sorry, I know this is a bad joke. So, what can I do for you now?"
        history_temp.append(response)

    else:
        session.attributes["headline"] = ''.join([i if ord(i) < 128 else ' ' for i in data['headline']])
        session.attributes["body"] = ''.join([i if ord(i) < 128 else ' ' for i in data["body"]])
        config.NEXT_INTENTION = "body"
        response = "here is the news headline. {}. Would you like the details of the news?".format(
            session.attributes["headline"])
        history_temp.append(response)

    config.HISTORY.append(history_temp)
    return response


@ask.intent("NewsInput")
def newsComponent(topics, dates, usplaces, regions, cities):

    config.CURRENT_SESSION = request["intent"]["name"]
    global_param_reset()

    config.CURRENT_PARAMS.append(topics)
    config.CURRENT_PARAMS.append(dates)
    config.CURRENT_PARAMS.append(usplaces)
    config.CURRENT_PARAMS.append(regions)
    config.CURRENT_PARAMS.append(cities)

    if config.CURRENT_SESSION != config.PREVIOUS_SESSION or config.PREVIOUS_PARAMS != config.CURRENT_PARAMS:
        config.PREVIOUS_SESSION = config.CURRENT_SESSION
        config.PREVIOUS_PARAMS = config.CURRENT_PARAMS

        global_content_reset()
        global_count_reset()

        search_result = search_news(topics, dates, usplaces, regions, cities)
        config.CURRENT_CONTENT = search_result

        response = news_response_construct()

    else:
        global_count()
        response = news_response_construct()

    if config.CURRENT_PARAMS[0]:
        return question(response)

    else:
        print colored(response, 'yellow')
        return question("are you asking about news?")


@ask.intent("WeatherInfo")
def weather_today(loc):

    config.CURRENT_SESSION = request["intent"]["name"]
    global_param_reset()
    config.CURRENT_PARAMS.append(loc)

    if config.CURRENT_SESSION != config.PREVIOUS_SESSION or config.PREVIOUS_PARAMS != config.CURRENT_PARAMS:
        config.PREVIOUS_SESSION = config.CURRENT_SESSION
        config.PREVIOUS_PARAMS = config.CURRENT_PARAMS

        if not loc:
            location = "atlanta"
        else:
            location = loc

        global_content_reset()
        global_count_reset()

        forcast = WeatherForecast(location)
        config.CURRENT_CONTENT = forcast.weatherInfo()
        config.NEXT_INTENTION = "next day"
        return question(config.CURRENT_CONTENT["current_info"] + ". would you like forecasts for the following day?")

    else:
        return question(config.CURRENT_CONTENT["current_info"] + ". would you like forecasts for the following day?")

@ask.intent("AMAZON.YesIntent")
def yes_specific_match():
    if config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "body":
        answer = config.CURRENT_CONTENT[config.COUNT]["_source"]["body"]
        config.NEXT_INTENTION = "comments"
        return question("here is the news details. {}. would you like to have somme comments".format(answer))
    elif config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "comments":
        answer = [cmt for i, cmt in enumerate(config.CURRENT_CONTENT[config.COUNT]["_source"]["comment"]) if i < 3 and len(config.CURRENT_CONTENT[config.COUNT]["_source"]["comment"]) != 0]
        config.NEXT_INTENTION = "more news"
        global_count()
        return question("here are some comments. {}. would you like to have more news".format(answer))
    elif config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "more news":
        news_response_construct()
    elif config.CURRENT_SESSION == "WeatherInfo" and config.NEXT_INTENTION == "next day":
        if config.COUNT <= 4:
            answer = config.CURRENT_CONTENT["forecasts_info"][config.COUNT]
            global_count()
            return question(answer + ". would you like further forecasts for another following day?")

@ask.intent("GeneralNoIntent")
def no_specific_match():
    if config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "body":
        answer = "OK, would you like some jokes instead?"
        config.NEXT_INTENTION = "jokes"
        return question(answer)
    elif config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "comments":
        answer = "OK, would you like some jokes instead?"
        config.NEXT_INTENTION = "jokes"
        return question(answer)
    elif config.CURRENT_SESSION == "NewsInput" and config.NEXT_INTENTION == "more news":
        answer = "OK, would you like some jokes instead?"
        config.NEXT_INTENTION = "jokes"
        return question(answer)
    elif config.CURRENT_SESSION == "WeatherInfo" and config.NEXT_INTENTION == "next day":

        return question("OK, would you like some jokes?")

# @ask.intent("Amazon.NoInTent")
# def no_specific_match(response):
#     key = session.attributes.keys()[0]
#     if key == "forecasts":
#         return question("OK, then. what can I do for you now")
#     elif key == "body":
#         return question("OK, then. what can I do for you now")

@ask.intent("GeneralUtterance")
def general(sentence):

    config.CURRENT_SESSION = request["intent"]["name"]
    global_param_reset()
    config.CURRENT_PARAMS.append(sentence)

    if config.CURRENT_SESSION != config.PREVIOUS_SESSION or config.PREVIOUS_PARAMS != config.CURRENT_PARAMS:
        config.PREVIOUS_SESSION = config.CURRENT_SESSION
        config.PREVIOUS_PARAMS = config.CURRENT_PARAMS
    response = chitchat.chat(sentence)
    config.CURRENT_CONTENT = response
     # print colored(response, 'green')
    response = ''.join([i if ord(i) < 128 else ' ' for i in response])
    # print "I have not been prepared well yet, so I am echoing you. {}".format(sentence)
    return question("{}".format(response))

@ask.intent("AMAZON.StopIntent")
def exit_intent():
    exit_message = "It was greating talking to you. Have a great one!"
    return statement(exit_message)

@ask.session_ended
def session_ended():
    return "", 500

if __name__ == '__main__':
    app.run(debug=True)
