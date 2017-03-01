from cornell_data import CornellData
import nltk
import pickle


wiki_dir = "/Users/ZW/Downloads/a5/wikipedia.txt"
conv_dir = "/Users/ZW/Downloads/a5/"

movie_lines_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zihao/temp_folder/movie_lines.txt'

movie_conversations_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zihao/temp_folder/movie_conversations.txt'

data_conversation_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_conversation_data.txt'
# data_sentence_dir = '/home/zwan438/temp_folder/chitchat_conversation_data.txt'
# data_sentence_dir = '/home/zihao/temp_folder/chitchat_conversation_data.txt'

cd = CornellData(movie_lines_filename, movie_conversations_filename)
conversations = cd.getConversations()

