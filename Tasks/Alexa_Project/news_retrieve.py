import json
from elasticsearch import Elasticsearch


class NewsRetrival(object):
    def __init__(self):
        self.es = Elasticsearch()  # use default of localhost, port 9200
        with open("./merge.json") as fo:
            self.doc = json.load(fo)
        for art in self.doc['docs']:
            self.es.index(index='news', doc_type='article', id=art['id'], body=art)

    def search(self, mQuery):
        q = True

        while q:
            # mQuery = raw_input("What do you what to ask?\n")
            print("Finish token :)\n")
            print("Start News Matching...\n")
            mQuery = unicode(mQuery, "utf-8")
            data = self.es.search(index="news", body={"query": {"bool": {"should": [{"match": {"headline": "{}".format(mQuery)}},
                                                                               {"match": {
                                                                                   "theme": "{}".format(mQuery)}}]}}})

            if data['hits']['hits']:
                print data['hits']['hits'][0]['_source']
                return data['hits']['hits'][0]['_source']
            else:
                return [None, "sorry, I don't know, would you like a video clip instead"]