import json
from itertools import count

import requests

LIMIT_ARTICLES = 100
LIMIT_TITLE_LENGTH = 200

with open('data/raw/articles/articles_metadata.json', 'r') as file:  # noqa
    articles = json.load(file)
    articles = articles[:LIMIT_ARTICLES]

counter = count(1)

for article in articles:
    print(f'Processing article {next(counter)}')
    pdf_link = article['pdf_url']
    pdf_content = requests.get(pdf_link).content
    title = (
        f'{article['title'][:LIMIT_TITLE_LENGTH]}...'
        if len(article['title']) > LIMIT_TITLE_LENGTH
        else article['title']
    )
    cleaned_title = (
        title.lower()
        .replace('\n', '')
        .replace(' ', '_')
        .replace('/', '_')
        .replace(',', '')
    )

    with open(f'data/raw/articles/scrapped/{cleaned_title}.pdf', 'wb') as file:
        file.write(pdf_content)
