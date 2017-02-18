import os

path = "/Users/ZW/Downloads/dialogs/"

files = os.listdir(path)

for fs in files:
    dirr = "".join([path, fs])
    if os.path.isdir(dirr):
        if not os.listdir(dirr):
            os.rmdir(dirr)
        else:
            print "For folder {}, the files are {} \n".format(dirr, os.listdir(dirr))
