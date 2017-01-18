import numpy as np
from cornell_data import CornellData


movie_line_fileName = "/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt"
movie_conversation_fileName = "/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt"
cd = CornellData(movie_line_fileName=movie_line_fileName, movie_conversation_fileName=movie_conversation_fileName)
lines = cd.loadLines()
conversations = cd.loadConversations()


