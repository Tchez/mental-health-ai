from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from mental_helth_ai.processing_raw_data.article_scraper import ScieloSpider

settings = get_project_settings()
settings.update({
    'HTTPCACHE_ENABLED': True,
    'HTTPCACHE_DIR': 'httpcache',
    'HTTPCACHE_EXPIRATION_SECS': 604800,
    'HTTPCACHE_IGNORE_HTTP_CODES': [500, 503, 504, 400, 403, 404, 408],
    'HTTPCACHE_STORAGE': 'scrapy.extensions.httpcache.FilesystemCacheStorage',  # noqa
})

process = CrawlerProcess(settings=settings)
process.crawl(ScieloSpider)
process.start()
