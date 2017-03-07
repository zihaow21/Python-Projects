from flask import Flask
from flask_ask import Ask, statement, question
from Tasks.Chatbot_Practice.chitchat_component import ChitChat
from news_retrieve import NewsRetrival


newsRetrieval = NewsRetrival()

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
    welcome_message = "Hello there, what's up?"
    return question(welcome_message)

@ask.intent("NewsInput")
def newsComponent(news, time, usplaces, regions, cities):
    news = news
    time = time
    usplaces = usplaces
    regions = regions
    cities = cities
    if news != None:
        newsRetrieval.search("{}, {}, {}, {}".format(news, time, usplaces, regions, cities))

    else:
        furtherAsk(news)
        if yesIntent():
            result = newsRetrieval.search("{}, {}, {}, {}".format(news, time, usplaces, regions, cities))
        else:
            return question("what can I do for you then")

    print "News inputs are {}, {}, {}, {}, {}".format(news, time, usplaces, regions, cities)

    return statement("here is the news headlines".format(result))

@ask.intent("GeneralUtterance")
def general(sentence):
    print "the general utterance is {}".format(sentence)
    return statement("{}".format(sentence))

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
    return question("Do you want to know about {}".format(info))

if __name__ == '__main__':
    app.run(debug=True)
