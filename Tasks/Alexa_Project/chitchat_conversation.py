from flask import Flask
from flask_ask import Ask, statement, question
import json
import requests
import unidecode


app = Flask(__name__)
ask = Ask(app, "/chitchat")

def chitchat():
    sess = requests.Session()
    