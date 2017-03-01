from pycorenlp.corenlp import *
import jsonrpclib
from simplejson import loads


server = jsonrpclib.Server("http://localhost:8080")
file_name = "/Users/ZW/Downloads/conll03.eng.trn.gold.tsv"
with open(file_name, "r") as f:
    token_doc = []
    temp = []
    for line in f:
        token_temp = line.split("\t")
        if token_temp[0] == "-DOCSTART-" or line == '\n':
            token_doc.append(" ".join(temp))
            temp = []
        else:
            temp.append(token_temp[0])

token_doc.pop(0)
for sentence in token_doc:
    result = loads(server.parse(sentence))
    print result

print result