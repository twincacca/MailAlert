# <<< 20250526 >>>
# pip install playwright
# playwright install


from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# === List of URLs you want to process ===
urls = [
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22bioinformatics%22,%22label%22:%22bioinformatics%22,%22order%22:0,%22_id%22:%22free-text_bioinformatics%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22bioinfo%22,%22label%22:%22bioinfo%22,%22order%22:0,%22_id%22:%22free-text_bioinfo%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22microbiology%22,%22label%22:%22microbiology%22,%22order%22:0,%22_id%22:%22free-text_microbiology%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22microbio%22,%22label%22:%22microbio%22,%22order%22:0,%22_id%22:%22free-text_microbio%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22mikrobio%22,%22label%22:%22mikrobio%22,%22order%22:0,%22_id%22:%22free-text_mikrobio%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22ecology%22,%22label%22:%22ecology%22,%22order%22:0,%22_id%22:%22free-text_ecology%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22ecolog%22,%22label%22:%22ecolog%22,%22order%22:0,%22_id%22:%22free-text_ecolog%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
    "https://www.job-room.ch/job-search?filter=eyJzb3J0IjoiREFURV9ERVNDIiwiZGlzcGxheVJlc3RyaWN0ZWQiOmZhbHNlLCJjb250cmFjdFR5cGUiOiJBTEwiLCJ3b3JrbG9hZFBlcmNlbnRhZ2VNaW4iOjEwLCJ3b3JrbG9hZFBlcmNlbnRhZ2VNYXgiOjEwMCwiY29tcGFueSI6bnVsbCwib25saW5lU2luY2UiOjYwLCJvY2N1cGF0aW9ucyI6W10sImtleXdvcmRzIjpbeyJ0eXBlIjoiZnJlZS10ZXh0IiwicGF5bG9hZCI6ImJpb2luZm9ybWF0aWNzICIsImxhYmVsIjoiYmlvaW5mb3JtYXRpY3MgIiwib3JkZXIiOjAsIl9pZCI6ImZyZWUtdGV4dF9iaW9pbmZvcm1hdGljcyAifSx7InR5cGUiOiJmcmVlLXRleHQiLCJwYXlsb2FkIjoiYmlvaW5mbyIsImxhYmVsIjoiYmlvaW5mbyIsIm9yZGVyIjowLCJfaWQiOiJmcmVlLXRleHRfYmlvaW5mbyJ9LHsidHlwZSI6ImZyZWUtdGV4dCIsInBheWxvYWQiOiJtaWNyb2Jpb2xvZ3kiLCJsYWJlbCI6Im1pY3JvYmlvbG9neSIsIm9yZGVyIjowLCJfaWQiOiJmcmVlLXRleHRfbWljcm9iaW9sb2d5In0seyJ0eXBlIjoiZnJlZS10ZXh0IiwicGF5bG9hZCI6Im1pY3JvYmlvIiwibGFiZWwiOiJtaWNyb2JpbyIsIm9yZGVyIjowLCJfaWQiOiJmcmVlLXRleHRfbWljcm9iaW8ifSx7InR5cGUiOiJmcmVlLXRleHQiLCJwYXlsb2FkIjoibWlrcm9iaW8iLCJsYWJlbCI6Im1pa3JvYmlvIiwib3JkZXIiOjAsIl9pZCI6ImZyZWUtdGV4dF9taWtyb2JpbyJ9LHsidHlwZSI6ImZyZWUtdGV4dCIsInBheWxvYWQiOiJlY29sb2d5IiwibGFiZWwiOiJlY29sb2d5Iiwib3JkZXIiOjAsIl9pZCI6ImZyZWUtdGV4dF9lY29sb2d5In0seyJ0eXBlIjoiZnJlZS10ZXh0IiwicGF5bG9hZCI6ImVjb2xvZyIsImxhYmVsIjoiZWNvbG9nIiwib3JkZXIiOjAsIl9pZCI6ImZyZWUtdGV4dF9lY29sb2cifSx7InR5cGUiOiJmcmVlLXRleHQiLCJwYXlsb2FkIjoiZGV2ZWxvcGVyIiwibGFiZWwiOiJkZXZlbG9wZXIiLCJvcmRlciI6MCwiX2lkIjoiZnJlZS10ZXh0X2RldmVsb3BlciJ9LHsidHlwZSI6ImZyZWUtdGV4dCIsInBheWxvYWQiOiJsaW51eCIsImxhYmVsIjoibGludXgiLCJvcmRlciI6MCwiX2lkIjoiZnJlZS10ZXh0X2xpbnV4In0seyJ0eXBlIjoiZnJlZS10ZXh0IiwicGF5bG9hZCI6ImFybWVlIiwibGFiZWwiOiJhcm1lZSIsIm9yZGVyIjowLCJfaWQiOiJmcmVlLXRleHRfYXJtZWUifV0sImxvY2FsaXRpZXMiOltdLCJyYWRpdXMiOjMwfQ%3D%3D"
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22developer%22,%22label%22:%22developer%22,%22order%22:0,%22_id%22:%22free-text_developer%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22linux%22,%22label%22:%22linux%22,%22order%22:0,%22_id%22:%22free-text_linux%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D",
#    "https://www.job-room.ch/job-search?query-values=%7B%22occupations%22:%5B%5D,%22keywords%22:%5B%7B%22type%22:%22free-text%22,%22payload%22:%22armee%22,%22label%22:%22armee%22,%22order%22:0,%22_id%22:%22free-text_armee%22%7D%5D,%22localities%22:%5B%5D,%22radius%22:30%7D"
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



