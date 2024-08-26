import json
import re
from datetime import datetime
from itertools import count
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import scrapy
from rich import print

MINIMUM_YEAR = 1900


class ScieloSpider(scrapy.Spider):
    name = 'scielo'
    allowed_domains = ['search.scielo.org', 'scielo.br']
    start_urls = [
        'https://search.scielo.org/?q=sa%C3%BAde+mental&lang=pt&filter%5Bin%5D%5B%5D=scl'
    ]

    def __init__(self):
        self.articles_metadata = []

    def parse(self, response):
        try:
            articles = response.xpath('//div[@class="item"]')
            counter = count(1)

            for article in articles:
                metadata = {}
                title = article.xpath(
                    './/a/strong[@class="title"]/text()'
                ).get()
                link = article.xpath(
                    './/a/strong[@class="title"]/../@href'
                ).get()
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

                date = self.extract_valid_year(filtered_source_text)

                metadata['title'] = self.clean_text(title) if title else None
                metadata['link'] = link.strip() if link else None
                metadata['pdf_url'] = pdf_url.strip() if pdf_url else None
                metadata['source'] = 'SciELO'
                metadata['description'] = (
                    self.clean_text(description) if description else None
                )
                metadata['date'] = date

                print(f"{next(counter)} - {metadata['title']}")

                if link and not link.startswith('javascript:'):
                    self.articles_metadata.append(metadata)

            next_page_button = response.css('a.pageNext::attr(href)').get()
            print(f'Próxima página: {next_page_button}')
            if next_page_button and 'javascript:' in next_page_button:
                page_number = self.extract_page_number(next_page_button)
                if page_number:
                    next_page_url = self.build_next_page_url(
                        response.url, page_number
                    )
                    yield scrapy.Request(next_page_url, callback=self.parse)
            else:
                self.save_metadata()

        except Exception as e:
            print(f'Erro ao processar a página: {response.url}')
            print(e)

        finally:
            self.save_metadata()

    def save_metadata(self):
        print(f'Salvando {len(self.articles_metadata)} artigos.')
        with open(
            'data/raw/articles/new_articles_metadata.json', 'w', encoding='utf-8'
        ) as f:
            json.dump(self.articles_metadata, f, ensure_ascii=False, indent=4)

    @staticmethod
    def extract_valid_year(text_list):
        """Extrai e valida a primeira data encontrada na lista."""
        for text in text_list:
            if re.match(r'\d{4}', text):
                year = int(text.strip()[:4])
                if MINIMUM_YEAR <= year <= datetime.now().year:
                    return str(year)
        return None

    @staticmethod
    def extract_page_number(js_function):
        """Extrai o número da página da função JavaScript."""
        match = re.search(r"go_to_page\('(\d+)'\)", js_function)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def build_next_page_url(current_url, page_number):
        """Constrói a URL da próxima página baseada no número da página."""
        parsed_url = urlparse(current_url)
        query_params = parse_qs(parsed_url.query)
        query_params['page'] = [page_number]
        from_value = (int(page_number) - 1) * int(
            query_params.get('count', [15])[0]
        ) + 1
        query_params['from'] = [str(from_value)]
        new_query_string = urlencode(query_params, doseq=True)
        next_page_url = urlunparse(parsed_url._replace(query=new_query_string))
        return next_page_url

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,:;?!-]', '', text)
        return text.strip()
