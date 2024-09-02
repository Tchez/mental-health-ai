import requests
import json

LIMIT_ARTICLES = 100

with open('data/raw/articles/articles_metadata.json', 'r') as file:
    articles = json.load(file)
    articles = articles[:LIMIT_ARTICLES]

for article in articles:
    pdf_link = article['pdf_url']
    pdf_content = requests.get(pdf_link).content
    title = f'{article['title'][:150]}...' if len(article['title']) > 150 else article['title']
    cleaned_title = title.lower().replace('\n', '').replace(' ', '_').replace('/', '_')

    with open(
        f'data/raw/articles/scrapped/{cleaned_title}.pdf', 'wb'
    ) as file:
        file.write(pdf_content)
