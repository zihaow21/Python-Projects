import requests
import urllib2
import json
import pickle


start_date = "2017/02/12"
end_date = "2017/02/13"
count = "100"

News_API = "https://docs.washpost.com/docs?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"
Comment_API = "https://docs.washpost.com/comments?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"

response_news = requests.get(News_API)
response_comments = requests.get(Comment_API)

news_data = json.loads(response_news.content)
comments_data = response_comments

with open("/Users/ZW/dropbox/WashingtonPostData/NewsSample.txt", "w") as f:
# with open("/home/zwan438/WashingtonPost/NewsSample.txt", "w") as f:
    json.dump(news_data, f)
    #print >> f, news_data
