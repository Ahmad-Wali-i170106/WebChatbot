import datetime
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Comment
from loguru import logger


class UrlCrawlerException(Exception):
    pass


class UrlCrawler:
    def __init__(
        self,
        base_url,
        user_agent: str | None = None,
        timeout: int = 10,
        encoding: str | None = None,
        autoset_encoding: bool = True,
        headers: dict | None = None,
        proxies: dict | None = None,
        crawl_delay: float = 10.0,
    ):
        self.base_url = self._normalize_url(base_url)
        self.session = requests.Session()
        # self.user_agent = user_agent or (
        #     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        #     "AppleWebKit/537.36 (KHTML, like Gecko) "
        #     "Chrome/122.0.0.0 Safari/537.36"
        # )
        self.encoding = encoding
        self.autoset_encoding = autoset_encoding
        self.proxies = proxies
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        self.timeout = timeout
        self.crawl_delay = crawl_delay
        if headers:
            self.session.headers.update(headers)
        self.session.headers.update({"User-Agent": self.user_agent})
        self.robot_parser = None
        self.base_netloc = urlparse(self.base_url).netloc

    def _normalize_url(self, url: str) -> str:
        """Standardize URLs to avoid redundant crawling."""
        parsed = urlparse(url)
        # Lowercase scheme/netloc, remove fragments, strip trailing slash from path
        path = parsed.path.rstrip("/")
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                parsed.params,
                parsed.query,
                "",
            )
        )

    def _parse_robots_txt(self):
        parsed = urlparse(self.base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(robots_url)
        try:
            self.robot_parser.read()
            delay = self.robot_parser.crawl_delay(
                self.user_agent
            ) or self.robot_parser.crawl_delay("*")
            if delay:
                self.crawl_delay = float(delay)
        except Exception:
            pass

    def _is_allowed(self, url):
        if not self.robot_parser:
            self._parse_robots_txt()
        return self.robot_parser.can_fetch(self.user_agent, url)  # type: ignore

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extracts visible text optimized for LLM readability."""
        for element in soup(
            ["script", "style", "head", "title", "meta", "[document]", "footer", "nav"]
        ):
            element.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        content_area = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"id": "content"})
            or soup
        )
        text = content_area.get_text(separator="\n", strip=True)
        return re.sub(r"\n+", "\n", text).strip()

    def _extract_metadata(
        self, soup: BeautifulSoup, url: str, resp: requests.Response
    ) -> dict:
        metadata = {
            "source": url,
            "timestamp": datetime.datetime.now().isoformat(),
            "content_type": resp.headers.get("Content-Type", ""),
        }
        if title := soup.find("title"):
            metadata["title"] = (
                title.get_text(strip=True)
                .strip()
                .replace("<title>", "")
                .replace("</title>", "")
            )
        else:

            def create_title_from_url(url: str) -> str:
                par_url = urlparse(url)
                return f"{par_url.netloc.lower()}-{par_url.path.lower().strip()}"

            metadata["title"] = (
                soup.title.string.strip()
                if soup.title and soup.title.string
                else create_title_from_url(url)
            )

        if desc := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = str(desc.get("content", "")).strip()
        if html := soup.find("html"):
            metadata["language"] = str(html.get("lang", "en"))
        return metadata

    def _is_pagination(self, url: str) -> bool:
        """Detects if a URL is likely a pagination link, including dynamic e-commerce patterns."""
        pagination_patterns = [
            r"[?&][^=]*page[^=]*=\d+",
            r"[?&]p=\d+",
            r"/page/\d+",
            r"/p/\d+",
        ]
        return any(re.search(pattern, url, re.I) for pattern in pagination_patterns)

    def _get_sitemap_urls(self) -> list[str]:
        parsed = urlparse(self.base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = self.session.get(robots_url)
        sitemap_urls = []
        for line in resp.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_urls.append(line.split(":", 1)[1].strip())
        all_urls: set[str] = set()
        for sitemap_url in sitemap_urls:
            try:
                sitemap_resp = self.session.get(sitemap_url)
                tree = ET.fromstring(sitemap_resp.content)
                for url in tree.findall(".//{*}loc"):
                    loc = url.text.strip() if url.text else ""
                    if loc and self._is_allowed(loc):
                        all_urls.add(loc)
            except Exception:
                continue
        return list(all_urls)

    def crawl(self, max_depth: int = 2):
        self._parse_robots_txt()
        if sitemap_urls := self._get_sitemap_urls():
            queue = deque([(self._normalize_url(url), 0) for url in sitemap_urls])
        else:
            queue = deque([(self.base_url, 0)])

        visited = set()
        results = []

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            # if not self._is_allowed(url):
            #     continue

            try:
                resp = self.session.get(url, timeout=self.timeout, proxies=self.proxies)
                if self.encoding is not None:
                    resp.encoding = self.encoding
                elif self.autoset_encoding:
                    resp.encoding = resp.apparent_encoding

                resp.raise_for_status()

                if any(
                    term in resp.text.lower()
                    for term in ["captcha", "bot protection", "sgcaptcha"]
                ):
                    raise UrlCrawlerException(f"Blocked by anti-bot measures at {url}")

                soup = BeautifulSoup(resp.text, "html.parser")
                metadata = self._extract_metadata(soup, url, resp)
                text = self._extract_text(soup)
                results.append(
                    {
                        "source": url,
                        "text": text,
                        "metadata": metadata,
                    }
                )
                visited.add(url)

                for link in set(soup.find_all("a", href=True)):
                    abs_url = self._normalize_url(urljoin(url, link["href"]))  # type: ignore
                    parsed_abs = urlparse(abs_url)

                    if parsed_abs.netloc.replace(
                        "www.", ""
                    ) == self.base_netloc.replace("www.", ""):
                        if (
                            abs_url not in visited
                            and (abs_url, depth) not in queue
                            and self._is_allowed(abs_url)
                        ):
                            if not re.search(
                                r"\.(pdf|jpg|png|docx?|zip|css|js|ico|gif|svg|csv|bz2|epub|webp|xlsx?|pptx?|pptm)$",
                                abs_url,
                                re.I,
                            ):
                                next_depth = (
                                    depth if self._is_pagination(abs_url) else depth + 1
                                )
                                queue.append((abs_url, next_depth))

                time.sleep(self.crawl_delay)
            except Exception as e:
                # results.append(
                #     {
                #         "text": f"Error scraping: {e}",
                #         "metadata": {"error": True, "source": url},
                #     }
                # )
                logger.error(f"Error scraping {url}: {e}")
                continue
        return results

    def lazy_crawl(self, max_depth: int = 2):
        self._parse_robots_txt()
        if sitemap_urls := self._get_sitemap_urls():
            queue = deque([(self._normalize_url(url), 0) for url in sitemap_urls])
        else:
            queue = deque([(self.base_url, 0)])

        visited = set()
        results = []

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            # if not self._is_allowed(url):
            #     continue

            try:
                resp = self.session.get(url, timeout=self.timeout, proxies=self.proxies)
                if self.encoding is not None:
                    resp.encoding = self.encoding
                elif self.autoset_encoding:
                    resp.encoding = resp.apparent_encoding

                resp.raise_for_status()

                if any(
                    term in resp.text.lower()
                    for term in ["captcha", "bot protection", "sgcaptcha"]
                ):
                    raise UrlCrawlerException(f"Blocked by anti-bot measures at {url}")

                soup = BeautifulSoup(resp.text, "html.parser")
                metadata = self._extract_metadata(soup, url, resp)
                text = self._extract_text(soup)
                yield {
                    "source": url,
                    "text": text,
                    "metadata": metadata,
                }
                visited.add(url)

                for link in set(soup.find_all("a", href=True)):
                    abs_url = self._normalize_url(urljoin(url, link["href"]))  # type: ignore
                    parsed_abs = urlparse(abs_url)

                    if parsed_abs.netloc.replace(
                        "www.", ""
                    ) == self.base_netloc.replace("www.", ""):
                        if (
                            abs_url not in visited
                            and (abs_url, depth) not in queue
                            and self._is_allowed(abs_url)
                        ):
                            if not re.search(
                                r"\.(pdf|jpg|png|docx?|zip|css|js)$", abs_url, re.I
                            ):
                                next_depth = (
                                    depth if self._is_pagination(abs_url) else depth + 1
                                )
                                queue.append((abs_url, next_depth))

                time.sleep(self.crawl_delay)
            except Exception as e:
                # results.append(
                #     {
                #         "text": f"Error scraping: {e}",
                #         "metadata": {"error": True, "source": url},
                #     }
                # )
                logger.error(f"Error scraping {url}: {e}")
                continue
        return results


if __name__ == "__main__":
    # Usage
    crawler = UrlCrawler("https://www.funavry.com")
    try:
        data = crawler.crawl(max_depth=3)
        for item in data:
            print(
                f"URL: {item['metadata']['source']}\nContent: {item['text'][:200]}...\n"
            )
    except UrlCrawlerException as e:
        print(f"Scraping stopped: {e}")
