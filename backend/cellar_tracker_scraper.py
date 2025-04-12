from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# Function to get reviews from a CellarTracker list page using Selenium
def get_reviews_from_list_page(url, max_pages=40):
    # Initialize WebDriver (Chrome)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open the webpage
    driver.get(url)

    # Wait for the page to load
    time.sleep(20)

    # Create an empty list to store reviews
    reviews = []

    # Track the current page number
    current_page = 1

    # Loop to extract reviews and handle pagination, stop after max_pages
    while current_page <= max_pages:
        # Check for CAPTCHA presence randomly, pause if CAPTCHA is found
        try:
            # Check if CAPTCHA is present by finding the container with class 'captcha-container'
            captcha_element = driver.find_element(By.ID, 'captcha-container')
            print("CAPTCHA detected! Please complete it and press Enter to continue.")
            input("Press Enter after completing the CAPTCHA.")
        except:
            # No CAPTCHA found, continue scraping
            pass
        
        # Loop through each review entry (check if there is a 'Notes' section on the page)
        rows = driver.find_elements(By.XPATH, "//tbody/tr[position()>1]")

        print(f"rows: {len(rows)}")

        # Extract details for each wine entry
        for row in rows:
            try:
                # Extract the wine name (inside <h3> in <span class="el nam"> within the <td class="name">)
                wine_name = row.find_element(By.XPATH, ".//td[@class='name']//h3").text.strip()

                # Extract the variety (inside <span class='el var'>)
                variety = row.find_element(By.XPATH, ".//td[@class='name']//span[@class='el var']").text.strip()

                # Extract the review text (inside <p class="break_word">) in the sibling <td>
                review_text = row.find_element(By.XPATH, ".//td[@class='score break_word_wrapper']//p[@class='break_word']").text.strip()

                # Extract the rating (inside <h3>) in the sibling <td>
                rating = row.find_element(By.XPATH, ".//td[@class='score break_word_wrapper']//h3").text.strip()

                # Print the extracted data
                print(f"Wine Name: {wine_name}")
                # print(f"Location: {location}")
                print(f"Variety: {variety}")
                print(f"Review: {review_text}")
                print(f"Rating: {rating}")
                print("-" * 40)
                reviews.append({
                    'Wine Name': wine_name,
                    'Review': review_text,
                    'Rating': rating,
                    'Variety': variety
                })
            except Exception as e:
                print(f"Error processing a wine entry: {e}")
        try:
            # Look for the "Next" button and click it
            
            next_button = driver.find_element(By.XPATH, '//a[contains(text(),"Next")]')
            next_button.click()

            # Wait for the page to load
            time.sleep(3)

            # Increment the page counter
            current_page += 1
            print(f"Scraping page {current_page}")
        except:
            # If there's no "Next" button (i.e., we've reached the last page), break out of the loop
            print("No more pages. Scraping complete.")
            break

    # Close the WebDriver
    driver.quit()

    return reviews

# URL of the CellarTracker list page
url = 'https://www.cellartracker.com/list.asp?table=Notes&iUserOverride=0&T=1000#selected%3DW4820241_1_K203cfff5f42281a9e7d7f87aa40f9414'

# Get reviews from the page, with a limit of 40 pages
print(f"Scraping reviews from: {url}")
wine_reviews = get_reviews_from_list_page(url, max_pages=40)

# Check if any reviews were scraped
print(f"Total reviews scraped: {len(wine_reviews)}")

# If reviews exist, save them to CSV
if wine_reviews:
    # Convert reviews to DataFrame
    df = pd.DataFrame(wine_reviews)

    # Save to CSV file
    df.to_csv('cellartracker_wine_reviews_with_rating_40pages.csv', index=False)
    print("Scraping complete! Data saved to 'cellartracker_wine_reviews_with_rating_40pages.csv'")
else:
    print("No reviews scraped.")
