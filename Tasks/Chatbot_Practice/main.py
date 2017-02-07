from tqdm import tqdm
from cornell_data import CornellData
from data_utils import DataUtils


movie_lines_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt'
movie_conversations_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt'

cd = CornellData(movie_lines_filename, movie_conversations_filename)
conversations = cd.getConversations()

du = DataUtils(conversations, 20, 20, 20, 100)  # DataUtils(conversations, maxLength, maxLengthEnco, maxLengthDeco, batchSize)

epochs = 10
for i in range(epochs):
    batch = du.getBatches()

