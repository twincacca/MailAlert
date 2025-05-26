# <<< 20250526 >>>
# pip install playwright
# playwright install


from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# === List of URLs you want to process ===
urls = [
#     "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22developer%22,%22label%22:%22developer%22,%22order%22:0,%22_id%22:%22free-text_developer%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22ecology%22,%22label%22:%22ecology%22,%22order%22:0,%22_id%22:%22free-text_ecology%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22bioinformatics%22,%22label%22:%22bioinformatics%22,%22order%22:0,%22_id%22:%22free-text_bioinformatics%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D"
]

# === Output file ===
output_file = "all_jobs.txt"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    with open(output_file, "w", encoding="utf-8") as out:
        for url in urls:
            print(f"\nProcessing: {url}")
            page.goto(url)
            page.wait_for_timeout(5000)  # wait for JS

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Extract job titles
            titles = soup.find_all(attrs={"data-test": "resultTitle"})

            for title in titles:
                job_title = title.get_text(strip=True)
                company = title.find_next("span", class_="subtitle")
                company_name = company.get_text(strip=True) if company else "N/A"

            #     print(f"Job Title: {job_title}")
            #     print(f"Company:   {company_name}")
            #     print("-" * 40)
                print(f"{job_title} || {company_name}")

    browser.close()



