import requests
import os
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pdfplumber
from typing import List
from dotenv import load_dotenv


class CrustPDF:
    def __init__(self, download_dir="pdfs"):
        load_dotenv()
        self.token = os.getenv("CRUST_API_KEY")
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    # -----------------------------
    # 1. SEARCH (Crust / any API)
    # -----------------------------
    def search(self, query: str) -> list[str]:
        url = "https://api.crustdata.com/web/search/live"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "x-api-version": "2025-11-01"
        }

        payload = {
            "query": query,
            "location": "US",
            "sources": ["ai","web"]
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()

            results = data.get("results", [])

            links = []
            for item in results:
                # Try common fields
                if "url" in item:
                    links.append(item["url"])
                elif "link" in item:
                    links.append(item["link"])

            return links
        except Exception as e:
            print("Search error:", e)
            return []    
    
    # 2. FILTER PDF LINKS
    # -----------------------------
    def filter_pdfs(self, urls: List[str]) -> List[str]:
        pdf_links = []
        for url in urls:
            if url.lower().endswith(".pdf"):
                pdf_links.append(url)
            # else:
            #     pdf_links.extend(self._extract_pdf_from_page(url))

        return list(set(pdf_links))

    def _extract_pdf_from_page(self, url: str) -> List[str]:
        pdf_links = []

        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower():
                    full_url = urljoin(url, href)
                    pdf_links.append(full_url)

        except Exception as e:
            print("Page scrape error:", e)

        return pdf_links

    # -----------------------------
    # 3. DOWNLOAD PDF
    # -----------------------------
    def download_pdf(self, url: str) -> str:
        try:
            filename = os.path.join(
                self.download_dir,
                os.path.basename(url.split("?")[0]) or f"file_{hash(url)}.pdf"
            )

            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            print(f"Downloaded: {filename}")
            return filename

        except Exception as e:
            print("Download error:", e)
            return None

    # -----------------------------
    # 4. EXTRACT TEXT
    # -----------------------------
    def extract_text(self, pdf_path: str) -> str:
        text = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        except Exception as e:
            print("Extraction error:", e)

        return text

    # -----------------------------
    # 5. FULL PIPELINE
    # -----------------------------
    def run(self, query: str):
        print(f"Searching for: {query}")

        urls = self.search(query)
        pdf_links = self.filter_pdfs(urls)

        print(f"Found {len(pdf_links)} PDFs")

        results = []

        for pdf_url in pdf_links:
            pdf_path = self.download_pdf(pdf_url)
            if pdf_path:
                text = self.extract_text(pdf_path)
                print(text)
                results.append({
                    "url": pdf_url,
                    "file": pdf_path,
                })
            break

        return results


# -----------------------------
# USAGE
# -----------------------------
# if __name__ == "__main__":
#     crawler = CrustPDF()

#     results = crawler.run("HP Laser MFP 323sdnw User Guide Manual Pdf")

#     for r in results:
#         print("\n--- PDF ---")
#         print("URL:", r["url"])
#         print("Preview:", r["text"][:500])