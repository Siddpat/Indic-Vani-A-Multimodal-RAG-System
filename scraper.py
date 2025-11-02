import requests
from bs4 import BeautifulSoup
import sys



def scrape_wikipedia_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

        print(f"Fetching article from {url}...")
        response = requests.get(url, headers=headers)
        
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')

        content_div = soup.find(id='mw-content-text')
        title = soup.find(id='firstHeading').text

        if not content_div:
            print("Error: Could not find the main content div 'mw-content-text'.")
            return

        paragraphs = content_div.find_all('p')
        article_text = "\n".join([p.get_text() for p in paragraphs])

        file_name = f"{title.replace(' ', '_')}.txt"
        
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(article_text)

        print(f"Successfully scraped and saved to '{file_name}'")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    url_to_scrape = "https://hi.wikipedia.org/wiki/%E0%A4%AE%E0%A5%81%E0%A4%97%E0%A4%BC%E0%A4%B2_%E0%A4%B5%E0%A4%BE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%81%E0%A4%95%E0%A4%B2%E0%A4%BE"
    scrape_wikipedia_article(url_to_scrape)



