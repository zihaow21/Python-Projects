from elasticsearch import Elasticsearch
import os
import json


class LoadElasticSearch(object):
    def __init__(self):
        self.es = Elasticsearch()  # use default of localhost, port 9200
        self.path = "2017/"
        for filename in os.listdir(self.path):
            with open(self.path + filename, 'r') as fo:
                doc = json.load(fo)

            for art in doc['docs']:
                self.es.index(index='news', doc_type='article', body=art)

# loadElasticSearch = LoadElasticSearch()