from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time

# URL restricted to Ticino + Italian
url = "https://www.ornitho.ch/index.php?m_id=4&p_c=duration&p_cc=-&sp_tg=1&sp_DateSynth=12.08.2025&sp_DChoice=offset&sp_DOffset=2&sp_SChoice=category&sp_Cat[never]=1&sp_Cat[veryrare]=1&sp_Cat[rare]=1&sp_Cat[unusual]=1&sp_Cat[escaped]=1&sp_Cat[common]=1&sp_Cat[verycommon]=1&sp_cC=-000800000&sp_FChoice=list&sp_FGraphFormat=auto&sp_FMapFormat=none&sp_FDisplay=DATE_PLACE_SPECIES&sp_FOrder=ALPHA&sp_FOrderListSpecies=ALPHA&sp_FListSpeciesChoice=DATA&sp_FOrderSynth=ALPHA&sp_FGraphChoice=DATA&sp_DFormat=DESC&sp_FAltScale=250&sp_FAltChoice=DATA&sp_FExportFormat=XLS&langu=it"

output_file = "all_birds.txt"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Change to True for silent run
    page = browser.new_page()
    page.goto(url)

    # Scroll until no new content appears
    last_height = None
    while True:
        # Get current page height
        curr_height = page.evaluate("document.body.scrollHeight")
        if last_height == curr_height:
            break
        last_height = curr_height

        # Scroll to bottom
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)  # small delay for new content to load

    # Now all content should be loaded
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    with open(output_file, "w", encoding="utf-8") as out:
        current_region = None
        first_region = True

        for elem in soup.find_all(["a", "span"]):
            # Detect region headings
            if elem.name == "a" and "login" in elem.get("href", ""):
                if not first_region:
                    out.write("-" * 40 + "\n\n")
                first_region = False

                current_region = elem.get_text(strip=True)
                out.write(f"Region: {current_region}\n")

            # Detect bird species
            elif elem.name == "span" and "sighting_detail" in elem.get("class", []):
                common_name_tag = elem.find("b")
                if not common_name_tag:
                    continue

                sci_name_tag = elem.find("span", class_="sci_name")
                common_name = common_name_tag.get_text(strip=True)
                sci_name = sci_name_tag.get_text(strip=True) if sci_name_tag else ""

                out.write(f"  - {common_name} {sci_name}\n")

    browser.close()

print("✅ Bird data saved to all_birds.txt")
