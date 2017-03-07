import json
import re


fileName = '/Users/ZW/Downloads/cache-1000000-json'
# new_data_file = '/Users/ZW/Dropbox/data/political_train_utterance.txt'
new_data_file = '/Users/ZW/Dropbox/data/political_test_utterance.txt'
with open(fileName, 'r') as f:
    data = f.readlines()

new_data = []
for i, item in enumerate(data):
    # if i< 8000:
    if i > 8000 and i < 16000:
        item = json.loads(item)
        text = ' '.join(re.sub("(RT )|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", item['text']).split())
        new_data.append(text)

with open(new_data_file, 'w') as f:
    for l in new_data:
        print >> f, l
