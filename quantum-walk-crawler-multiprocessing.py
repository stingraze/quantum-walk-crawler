#Quantum Walk Web Crawler Ver. 0.1 - Created with help of Claude 3.5
#Referenced: Quantum walk based search algorithms - https://arxiv.org/abs/0808.0059
#Created by Tsubasa Kato - Last Updated: 7/11/2024 0:51PM JST
#My Website: https://www.tsubasakato.com 
#My Company Website: https://www.inspiresearch.io/en
#Experimental Version Still.
import random
import math
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import multiprocessing
from functools import partial
import sys

def get_page_content(url):
    """Fetch and return the text content of a web page."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        return url, ' '.join(soup.stripped_strings)
    except Exception as e:
        print(f"Failed to fetch content from {url}: {str(e)}")
        return url, ""

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

def crawl_page(url, visited_urls):
    if url not in visited_urls:
        print(f"Crawling: {url}")
        new_links = get_links(url)
        print(f"Found {len(new_links)} new links on {url}")
        return url, new_links
    return url, []

def quantum_walk_web_crawler(start_url, max_pages):
    print(f"Starting crawl from {start_url}")
    urls_to_visit = [start_url]
    visited_urls = set()
    
    # Collect URLs
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        while len(visited_urls) < max_pages and urls_to_visit:
            crawl_results = pool.map(partial(crawl_page, visited_urls=visited_urls), urls_to_visit[:min(len(urls_to_visit), max_pages - len(visited_urls))])
            for url, new_links in crawl_results:
                visited_urls.add(url)
                urls_to_visit.extend(new_links)
            urls_to_visit = list(set(urls_to_visit) - visited_urls)
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
        subset_urls = [urls[index] for index in subset]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            contents = dict(pool.map(get_page_content, subset_urls))
        
        # Check for duplicates
        content_dict = {}
        for url, content in contents.items():
            if content in content_dict:
                print("Duplicate content found!")
                return url, content_dict[content]
            content_dict[content] = url
        
        print("No duplicates found in this iteration")
        time.sleep(1)  # Be nice to the server
    
    # If no collision is found, return None
    print("\nSearch complete. No duplicate content found.")
    return None

if __name__ == "__main__":
    # Example usage
    start_url = sys.argv[1]
    max_pages = 100

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
