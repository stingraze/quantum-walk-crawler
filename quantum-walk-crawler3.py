#Quantum Walk Web Crawler - Created with help of Claude 3.5
#Referenced: Quantum walk based search algorithms - https://arxiv.org/abs/0808.0059
#Created by Tsubasa Kato 7/11/2024 0:36PM JST
#My Website: https://www.tsubasakato.com 
#My Company Website: https://www.inspiresearch.io/en
#Experimental Version Still.
import random
import math
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import sys

def get_page_content(url):
    """Fetch and return the text content of a web page."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        print(f"Failed to fetch content from {url}: {str(e)}")
        return ""

def get_links(url):
    """Extract all links from a web page."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        links = [urljoin(base_url, link.get('href')) for link in soup.find_all('a', href=True)]
        return [link for link in links if urlparse(link).netloc == urlparse(base_url).netloc]
    except Exception as e:
        print(f"Failed to fetch links from {url}: {str(e)}")
        return []

def quantum_walk_web_crawler(start_url, max_pages):
    """
    Simulates a quantum walk search for duplicate content across web pages.
    
    :param start_url: The starting URL for the crawl
    :param max_pages: Maximum number of pages to consider
    :return: A pair of URLs with duplicate content, or None if no such pair exists
    """
    print(f"Starting crawl from {start_url}")
    urls_to_visit = [start_url]
    visited_urls = set()
    
    # Collect URLs
    while len(visited_urls) < max_pages and urls_to_visit:
        url = urls_to_visit.pop(0)
        if url not in visited_urls:
            print(f"Crawling: {url}")
            visited_urls.add(url)
            new_links = get_links(url)
            print(f"Found {len(new_links)} new links")
            urls_to_visit.extend(new_links)
        
        print(f"Visited {len(visited_urls)} pages, {len(urls_to_visit)} pages in queue")
        time.sleep(1)  # Be nice to the server
    
    urls = list(visited_urls)
    n = len(urls)
    
    print(f"\nCrawling complete. Collected {n} unique URLs.")
    
    if n < 2:
        return None
    
    # Set the size of the subset (as per the optimal choice in the paper)
    r = min(int(n**(2/3)), n)
    
    # Number of iterations for the quantum walk (simplified)
    iterations = int(math.sqrt(n / r))
    
    print(f"\nStarting quantum-inspired walk search:")
    print(f"Subset size: {r}")
    print(f"Number of iterations: {iterations}")
    
    # Initialize the subset
    subset = random.sample(range(n), r)
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Simulate the quantum walk step
        old_index = random.randint(0, len(subset)-1)
        new_index = random.randint(0, n-1)
        if new_index not in subset:
            print(f"Updating subset: Replacing index {subset[old_index]} with {new_index}")
            subset[old_index] = new_index
        else:
            print("Subset unchanged in this iteration")
        
        # Check for a collision in the subset (simulating the checking step)
        print("Checking for duplicate content in the current subset")
        contents = {}
        for index in subset:
            print(f"Fetching content from {urls[index]}")
            content = get_page_content(urls[index])
            if content in contents:
                print("Duplicate content found!")
                return urls[index], urls[contents[content]]
            contents[content] = index
        
        print("No duplicates found in this iteration")
        time.sleep(1)  # Be nice to the server
    
    # If no collision is found, return None
    print("\nSearch complete. No duplicate content found.")
    return None

# Example usage
start_url = sys.argv[1]
max_pages = 20

print("Starting web crawler with quantum-inspired search algorithm")
print(f"Start URL: {start_url}")
print(f"Maximum pages to crawl: {max_pages}")

start_time = time.time()

result = quantum_walk_web_crawler(start_url, max_pages)

end_time = time.time()

if result:
    url1, url2 = result
    print(f"\nFound duplicate content on:\n{url1}\nand\n{url2}")
else:
    print("\nNo duplicate content found")

print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")