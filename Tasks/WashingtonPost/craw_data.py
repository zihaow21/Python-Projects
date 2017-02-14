import requests
import urllib2
import json
import pickle


start_date = "2017/02/12"
end_date = "2017/02/13"
count = "100"

News_API = "https://docs.washpost.com/docs?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"
Comment_API = "https://docs.washpost.com/comments?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"

response_news = requests.request("GET", News_API)
response_comments = requests.get(Comment_API)

news_data = response_news.content.decode('unicode_escape').encode('ascii','ignore')
comments_data = response_comments.content

with open("/Users/ZW/dropbox/WashingtonPostData/NewsSample.txt", "wb") as f:
    pickle.dump(news_data, f, protocol=pickle.HIGHEST_PROTOCOL)
