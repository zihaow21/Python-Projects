from multiprocessing import pool
import requests
import random
import bs4 as bs
import string


def random_starting_url():
    starting = ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(3))
    url = ''.join(['https://', starting, '.com'])

    return url


def handle_local_links(url, link):
    if link.startswith('/'):
        return ''.join([url, link])
    else:
        return link


def get_links(url):
