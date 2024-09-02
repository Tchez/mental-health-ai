import json
from itertools import count

import requests

LIMIT_ARTICLES = 100
LIMIT_TITLE_LENGTH = 200

with open('data/raw/articles/articles_metadata.json', 'r') as file:  # noqa
    articles = json.load(file)
    articles = articles[:LIMIT_ARTICLES]

counter = count(1)


def clear_title(title):
    title = (
        f'{title[:LIMIT_TITLE_LENGTH]}...'
        if len(title) > LIMIT_TITLE_LENGTH
        else title
    )

    cleaned_title = (
        title.lower()
        .replace('\n', '')
        .replace(' ', '_')
        .replace('/', '_')
        .replace(',', '')
    )
    return cleaned_title


for article in articles:
    print(f'Processing article {next(counter)}')
    pdf_link = article['pdf_url']
    pdf_content = requests.get(pdf_link).content
    cleaned_title = clear_title(article['title'])

    with open(f'data/raw/articles/scrapped/{cleaned_title}.pdf', 'wb') as file:
        file.write(pdf_content)
