from flask import Flask
from flask_ask import Ask, statement, question
from Tasks.Chatbot_Practice.chitchat_component import ChitChat
from news_retrieve import NewsRetrival
from weather_forcast import WeatherForecast


newsRetrieval = NewsRetrival()
# tokenizer = RegexpTokenizer(r'\w+')

app = Flask(__name__)
ask = Ask(app, "/emersonbot")

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
    return question(welcome_message)

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
            return statement(result)

        else:
            headlines = data['headline']
            headlines = ''.join([i if ord(i) < 128 else ' ' for i in headlines])
            body = data['body']
            body = ''.join([i if ord(i) < 128 else ' ' for i in body])
            return statement("here is the news headlines and details. {} {}".format(headlines, body))
    else:
        furtherAsk("the news")
        if yesIntent():
            data = newsRetrieval.search("{}, {}, {}, {}".format(news, time, usplaces, regions, cities))
            headlines = data['headline']
            headlines = ''.join([i if ord(i) < 128 else ' ' for i in headlines])
            body = data['body']
            body = ''.join([i if ord(i) < 128 else ' ' for i in body])
            return statement("here is the news headline and the details. {} {}".format(headlines, body))
        else:
            return question("what can I do for you then.")

@ask.intent("GeneralUtterance")
def general(sentence):
    print "I have been prepared well yet, so I am echoing you. {}".format(sentence)
    return statement("{}".format(sentence))

@ask.intent("AMAZON.SearchAction<object@WeatherForecast>")
def weather(object):
    location = object.location.addressLocality.name
    forcast = WeatherForecast(location)
    current_info, forecasts_info = forcast.weatherInfo()
    return statement(current_info)

@ask.intent("AMAZON.StopIntent")
def exit_intent():
    exit_message = "It was greating talking to you. Have a great one!"
    return statement(exit_message)

@ask.intent("AMAZON.YesIntent")
def yesIntent():
    return True

@ask.intent("AMAZON.NoIntent")
def noIntent():
    return False

def furtherAsk(info):
    return question("Do you want to know about {} or something else".format(info))



if __name__ == '__main__':
    app.run(debug=True)
