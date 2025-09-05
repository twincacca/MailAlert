from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time

# URL tutto CH per gru
url = "https://www.ornitho.ch/index.php?m_id=4&sp_DOffset=2&langu=it"
output_file = "all_birds.txt"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # headless for CI/GitHub Actions
    page = browser.new_page()
    page.goto(url)

    # Scroll until all lazy-loaded results are visible
    last_height = None
    while True:
        curr_height = page.evaluate("document.body.scrollHeight")
        if last_height == curr_height:
            break
        last_height = curr_height
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)

    # Parse final HTML
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    with open(output_file, "w", encoding="utf-8") as out:
        current_region = None
        first_region = True

        regions_found = False

        for elem in soup.find_all(["a", "span"]):
            # --- Case 1: Region headings exist ---
            if elem.name == "a" and "login" in elem.get("href", ""):
                regions_found = True
                if not first_region:
                    out.write("-" * 40 + "\n\n")
                first_region = False

                current_region = elem.get_text(strip=True)
                out.write(f"Region: {current_region}\n")

            # --- Case 2: Bird species ---
            elif elem.name == "span" and "sighting_detail" in elem.get("class", []):
                common_name_tag = elem.find("b")
                if not common_name_tag:
                    continue

                sci_name_tag = elem.find("span", class_="sci_name")
                common_name = common_name_tag.get_text(strip=True)
                sci_name = sci_name_tag.get_text(strip=True) if sci_name_tag else ""

                # If we have region headings: normal formatting
                if regions_found:
                    out.write(f"  - {common_name} {sci_name}\n")
                else:
                    # --- Case 2b: no headings, try to find location from parent row ---
                    parent_row = elem.find_parent(["tr", "div"])
                    location_text = None
                    if parent_row:
                        # Look for a <td> or <div> that is not the species itself
                        candidates = parent_row.find_all(text=True)
                        combined = " ".join(t.strip() for t in candidates if t.strip())
                        # Heuristic: species name is already known, remove it
                        location_text = combined.replace(common_name, "").strip()

                    out.write(f"{common_name} {sci_name} || Location: {location_text or 'N/A'}\n")

    browser.close()

print("✅ Bird data saved to all_birds.txt")
