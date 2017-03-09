import aiml
import os


kernel = aiml.Kernel()
sessionId = 12345

path_root = "templates/"

if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile = "bot_brain.brn")
else:
    kernel.bootstrap(learnFiles=path_root + "ai.aiml", commands = "load aiml b")
    kernel.saveBrain("bot_brain.brn")

# kernel now ready for use
while True:
    print kernel.respond(raw_input("Enter your message >> "), sessionId)