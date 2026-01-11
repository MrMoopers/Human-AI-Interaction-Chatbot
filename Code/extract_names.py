from bs4 import BeautifulSoup
import requests
import re

#attemptst o extract all names from wiki.

url = 'https://en.wikipedia.org/wiki/Tom_Lehrer'

r = requests.get(url)
namesSet = {''}

soup = BeautifulSoup(r.text, 'html.parser')
if 'born' in soup.text:
    x = soup.text.replace('\n', '')
    
    result = x[0:x.find(' - wikipedia')] 
    
    
print(namesSet)



# wikiName = [x.find_all('a') for x in soup.find_all('div', class_ = 'div-col columns column-count column-count-5')]
# for names in wikiName:
    
    
#     #infobox biography vcard
#     #born --> get title
#     print([name.text for name in names if name.text != 'wikt' and name.text != '@'])