---
title: BeautifulSoup Quick Start
date: 2022-04-22
categories:
  - Python
  - HTML
tags:
   - HTML
   - Web Crawling
   - BeautifulSoup
---


### Index_prac.html
```HTML
<!DOCTYPE html>

<html lang="en">

    <head>
        <title>
            The Dormouse's story
        </title>
    </head>


    <body>
        <p class="title"><b>The Dormouse's story</b></p>

        <p class="story">
            Once upon a time there were three little sisters; and their names were
            
                <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
                <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
                <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;

            and they lived at the bottom of a well.
        </p>

        <p class="story">To Be Continued...</p>
    </body>
</html>
```


### Quick Start.py
```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(open("index_prac.html"), 'html.parser')

# print all
print(soup.prettify())

# navigate that data structure
print(soup.title)
# <title>The Dormouse's story</title>

print(soup.title.name)
# u'title'

print(soup.title.string)
# u'The Dormouse's story'

print(soup.title.parent.name)
# u'head'

print(soup.p)
# <p class="title"><b>The Dormouse's story</b></p>

print(soup.p['class'])
# u'title'

print(soup.a)
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

print(soup.find_all('a'))
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

print(soup.find(id="link3"))
# <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

## extracting all the URLs
for link in soup.find_all('a'):
    print(link.get('href'))
# http://example.com/elsie
# http://example.com/lacie
# http://example.com/tillie

## extracting all the text from a page
print(soup.get_text())
```

---
- Reference: [https://www.crummy.com/software/BeautifulSoup/bs4/doc/#quick-start](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#quick-start)