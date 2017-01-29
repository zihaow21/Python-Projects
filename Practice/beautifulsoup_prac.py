import bs4 as bs
from urllib2 import urlopen
import pandas as pd




# url = 'https://pythonprogramming.net/sitemap.xml'
# sauce = urlopen(url).read()
# soup = bs.BeautifulSoup(sauce, 'xml')
# for url in soup.find_all('loc'):
#     print(url)

url = 'https://pythonprogramming.net/parsememcparseface'
sauce = urlopen(url).read()
soup = bs.BeautifulSoup(sauce, 'lxml')

# nav = soup.nav
# print(nav)
#
# for url in nav.find_all('a'):
#     print(url.get('href'))
#
# body = soup.body
# for paragraph in body.find_all('p'):
#     print(paragraph.text)
#
# for div in soup.find_all('div'):
#     print(div.text)
#
# for div in soup.find_all('div', class_='body'):
#     print(div.text)

# table = soup.find('table')
# table_rows = table.find_all('tr')
#
# for tr in table_rows:
#     td = tr.find_all('td')
#     row = [i.text for i in td]
#     print(row)

# dfs = pd.read_html('https://pythonprogramming.net/parsememcparseface', header=0)
# print dfs

