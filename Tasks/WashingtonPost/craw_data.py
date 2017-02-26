import requests
import json
import pymongo


# start_date = "2016/02/11"
# end_date = "2016/02/12"
# count = "10"
#
# News_API = "https://docs.washpost.com/docs?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"
# Comment_API = "https://docs.washpost.com/comments?stdate=" + start_date + "&enddate=" + end_date + "&offset=0&count=" + count + "&key=zgi53MiWSM9FbsYAPrvz"
#
# response_news = requests.get(News_API)
# response_comments = requests.get(Comment_API)
#
# news_data = json.loads(response_news.content.decode('ascii'))
# comments_data = json.loads(response_comments.content.decode('ascii'))
#
# keys = ["contenttype", "associatedbinaryurl"]
# for item1, item2 in zip(news_data['docs'], comments_data['docs']):
#     for key in keys:
#         item1.pop(key, None)
#
#     item1["comments"] = item2["comments"]
#     item1["date"] = start_date


connection = pymongo.MongoClient("mongodb://ali:alexanews@10.40.30.93:27017/washington_news")
db = connection.washington_news
record = db.collection

# for item in news_data["docs"]:
#     record.insert(item)

# with open("/Users/ZW/dropbox/WashingtonPostData/NewsSample.txt", "w") as f:
# # with open("/home/zwan438/WashingtonPost/NewsSample.txt", "w") as f:
#     json.dump(news_data, f)
#     # print >> f, news_data

