from elasticsearch import Elasticsearch
from termcolor import colored


class NewsRetrival(object):
    def __init__(self):
        self.es = Elasticsearch()

    def search(self, mQuery):
        # print colored(mQuery, 'red')
        mQuery = mQuery.replace("None", "")
        # print colored(mQuery, 'green')
        # mQuery = raw_input("What do you what to ask?\n")
        mQuery = unicode(mQuery, "utf-8")

        data = self.es.search(index="news", body={"query": {"bool": {"should": [{"match": {"headline": "{}".format(mQuery)}},{"match": {"body": "{}".format(mQuery)}},
                                                                               {"match": {"theme": "{}".format(mQuery)}}]}}})

        if data['hits']['hits']:
            print data['hits']['hits']
            return data['hits']['hits']
        else:
            return [None, "sorry, I don't know, would you like a video clip instead"]

# newsRetrieve = NewsRetrival()
# data = newsRetrieve.search("latest news, Barak Obama")
# print "the current search result is {}".format(data)