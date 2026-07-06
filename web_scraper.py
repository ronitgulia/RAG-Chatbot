"""
Web scraping module for RAG knowledge ingestion.
Supports both requests+BeautifulSoup and optional Selenium fallback.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import CONFIG

logger = logging.getLogger(__name__)


class WebScraper:
    """
    Scrape one or more URLs and return document dicts compatible with
    the rest of the RAG pipeline.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": CONFIG.scraping.user_agent})

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=CONFIG.scraping.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _parse_html(self, html: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "button", "iframe", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Prefer main content areas
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"id": re.compile(r"content|main", re.I)})
            or soup.find("div", {"class": re.compile(r"content|main|body", re.I)})
            or soup.body
        )

        text = self._clean_text(main.get_text(separator="\n") if main else "")

        # Collect all internal links for crawling
        links = []
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        from urllib.parse import urldefrag
        for a in soup.find_all("a", href=True):
            href, _ = urldefrag(urljoin(base, a["href"]))
            if self._is_valid_url(href) and href.startswith(base):
                links.append(href)

        return {
            "page_content": text,
            "metadata": {
                "source": url,
                "title": title,
                "file_type": "web",
                "char_count": len(text),
            },
            "links": list(set(links)),
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL and return a document dict."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        html = self._fetch_html(url)
        if html is None:
            return None
        doc = self._parse_html(html, url)
        logger.info(f"Scraped {url} — {len(doc['page_content'])} chars")
        return doc

    def scrape_multiple(
        self,
        urls: List[str],
        follow_links: bool = False,
        max_pages: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Scrape a list of URLs.
        If follow_links=True, also crawls discovered internal links
        up to max_pages total.
        """
        max_pages = max_pages or CONFIG.scraping.max_pages
        visited = set()
        queue = list(urls)
        documents = []

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            doc = self.scrape_url(url)
            if doc and doc["page_content"]:
                documents.append({
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                })
                if follow_links:
                    for link in doc.get("links", []):
                        if link not in visited and len(queue) < max_pages * 3:
                            queue.append(link)

        logger.info(f"Scraped {len(documents)} pages from {len(urls)} seed URL(s).")
        return documents

    def scrape_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fallback Selenium scraper for JavaScript-heavy pages.
        Requires selenium + chromedriver on PATH.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait

            opts = Options()
            opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-gpu")

            driver = webdriver.Chrome(options=opts)
            driver.set_page_load_timeout(CONFIG.scraping.timeout)
            driver.get(url)
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            html = driver.page_source
            driver.quit()

            doc = self._parse_html(html, url)
            logger.info(f"Selenium scraped {url}")
            return doc

        except Exception as e:
            logger.error(f"Selenium scraping failed for {url}: {e}")
            return None