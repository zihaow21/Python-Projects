#!/usr/bin/env python

"""
make-chatterbot
by Brandon Jackson
"""

import aiml
import subprocess
import os
import argparse
from MyKernel import MyKernel

BOT_PREDICATES = {
    "name": "KanoBot",
    "birthday": "January 1st 1969",
    "location": "London",
    "master": "Judoka",
    "website":"https://github.com/brandonjackson/make-chatterbot",
    "gender": "",
    "age": "",
    "size": "",
    "religion": "",
    "party": ""
}

DEVNULL = open(os.devnull, 'wb')

k = MyKernel()
 
# Load the AIML files on first load, and then save as "brain" for speedier startup
if os.path.isfile("cache/standard.brn") is False:
    k.learn("aiml/standard/std-startup.xml")
    k.respond("load aiml b")
    k.saveBrain("cache/standard.brn")
else:
    k.loadBrain("cache/standard.brn")
 
# Give the bot a name and lots of other properties
for key,val in BOT_PREDICATES.items():
    k.setBotPredicate(key, val)

# Start Infinite Loop
while True:
    # Prompt user for input
    input = raw_input("> ")

    # Send input to bot and print chatbot's response
    matchedPattern = k.matchedPattern(input) # note: this has to come before the 
                                             # call to respond as to reflect
                                             # the correct history
    response = k.respond(input)
    print response
    # if SHOW_MATCHES:
    #     print "Matched Pattern: "
    #     print k.formatMatchedPattern(matchedPattern[0])
    #     print "Response: "
    # print response
    #
    # # Output response as speech using espeak
    # if TTS_ENABLED is False:
    #     pass
    # elif TTS_ENGINE == "espeak":
    #     subprocess.call(["espeak", "-s", str(TTS_SPEED), "-v", TTS_VOICE,
    #                          "-p", str(TTS_PITCH), "\""+response+"\""],
    #                     stderr=DEVNULL)
    #
    # # Output response as speech using say
    # elif TTS_ENGINE == "say":
    #     args = ["say","-r", str(TTS_SPEED)]
    #     if TTS_VOICE_DEFAULT!=TTS_VOICE:
    #         args.append("-v")
    #         args.append(TTS_VOICE)
    #     args.append("\""+response+"\"")
    #     subprocess.call(args)
    #
    # # Output response as speech using unsupported TTS engine
    # else:
    #     subprocess.call([TTS_ENGINE, "\""+response+"\""])
