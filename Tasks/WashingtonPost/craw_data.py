import requests
import json


start_date = "2017/02/12"
end_date = "2017/02/13"
count = "10"

News_API = "https://docs.washpost.com/docs?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"
Comment_API = "https://docs.washpost.com/comments?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"

response_news = requests.get(News_API)
response_comments = requests.get(Comment_API)

news_data = json.loads(response_news.content.decode('ascii'))
comments_data = json.loads(response_comments.content.decode('ascii'))

keys = ["contenttype", "associatedbinaryurl"]
for item1, item2 in zip(news_data["docs"], comments_data["docs"]):
    for key in keys:
        item1.pop(key, None)

    item1["comments"] = item2["comments"]

# with open("/Users/ZW/dropbox/WashingtonPostData/NewsSample.txt", "w") as f:
with open("/home/zwan438/WashingtonPost/NewsSample.txt", "w") as f:
    json.dump(news_data, f)
    # print >> f, news_data

