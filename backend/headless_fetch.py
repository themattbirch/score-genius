# headless_fetch.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# --- Configuration ---
FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"
# Use headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# --- Initialize WebDriver ---
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 15)

try:
    print(f"Opening {FANGRAPHS_URL} ...")
    driver.get(FANGRAPHS_URL)
    
    # Wait a few seconds for page load
    time.sleep(5)
    
    # Scroll to the bottom to force lazy loading
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print("Scrolled to bottom; waiting for additional content to load...")
    time.sleep(5)
    
    # Use the new selector that targets the grid table
    css_selector = "#root-roster-resource .probables-grid .fg-data-grid .table-wrapper-inner table"
    print(f"Attempting to locate the grid table using CSS selector: {css_selector}")
    
    # Use WebDriverWait for the element to be present
    grid_table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))
    
    if grid_table:
        print("Successfully located the probables grid table!")
        # Optionally, save the page's HTML for debugging:
        with open("fangraphs_debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved debug HTML to 'fangraphs_debug.html'.")
    else:
        print("ERROR: Unable to locate the probables grid table in the rendered HTML.")
    
except Exception as e:
    print(f"ERROR: {e}")
finally:
    driver.quit()
