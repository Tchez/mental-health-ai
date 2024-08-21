import json
import scrapy
from rich import print


class ScieloSpider(scrapy.Spider):
    name = 'scielo'
    allowed_domains = ['search.scielo.org', 'scielo.br']
    start_urls = [
        'https://search.scielo.org/?q=sa%C3%BAde+mental&lang=pt&filter%5Bin%5D%5B%5D=scl'
    ]

    def __init__(self):
        self.articles_metadata = []

    def parse(self, response):
        articles = response.xpath('//div[@class="item"]')

        for article in articles:
            metadata = {}
            title = article.xpath('.//a/strong[@class="title"]/text()').get()
            link = article.xpath('.//a/strong[@class="title"]/../@href').get()
            pdf_url = article.xpath(
                './/a[contains(@href, "sci_pdf")]/@href'
            ).get()
            description = article.xpath(
                './/div[contains(@class, "abstract")]/text()'
            ).get()

            source_text_list = article.xpath(
                './/div[@class="line source"]//text()'
            ).getall()
            filtered_source_text = [
                text.strip() for text in source_text_list if text.strip()
            ]

            date = next(
                (
                    text
                    for text in filtered_source_text
                    if any(
                        year in text
                        for year in ['2024', '2023', '2022', '2021']
                    )
                ),
                None,
            )

            metadata['title'] = self.clean_text(title) if title else None
            metadata['link'] = link.strip() if link else None
            metadata['pdf_url'] = pdf_url.strip() if pdf_url else None
            metadata['source'] = 'SciELO'
            metadata['description'] = (
                self.clean_text(description) if description else None
            )
            metadata['date'] = date.strip()[:-1] if date else None

            if link and not link.startswith('javascript:'):
                self.articles_metadata.append(metadata)

        next_page = response.xpath('//a[@class="pageNext"]/@href').get()

        if next_page and not next_page.startswith('javascript:'):
            next_page_url = response.urljoin(next_page)
            yield scrapy.Request(next_page_url, callback=self.parse)
        else:
            self.save_metadata()

    def save_metadata(self):
        print(f"Salvando {len(self.articles_metadata)} artigos.")
        with open(
            'data/raw/articles/articles_metadata.json', 'w', encoding='utf-8'
        ) as f:
            json.dump(self.articles_metadata, f, ensure_ascii=False, indent=4)

    def clean_text(self, text):
        import re

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,:;?!-]', '', text)
        return text.strip()


# TODO: Ainda há bastante a ser feito para melhorar a qualidade dos dados coletados...
