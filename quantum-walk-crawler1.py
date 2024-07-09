# Quantum Walk-inspired Web Crawler
#(C) Tsubasa Kato created with help of Perplexity AI and ChatGPT (GPT-4o)
# Parameters
import requests
from bs4 import BeautifulSoup
import cmath
import random
import math
import sys
import time
# Quantum Walk-inspired Web Crawler

# Parameters
num_steps = 10
num_sites = 5
initial_url = sys.argv[1]
output_file = "crawled_urls.txt"

# Quantum state initialization
state = [cmath.rect(1 / math.sqrt(num_sites), 0) for _ in range(num_sites)]

# Hadamard coin operator
def hadamard_coin(state):
    new_state = []
    for i in range(len(state)):
        new_state.append((state[i] + state[(i + 1) % num_sites]) / cmath.sqrt(2))
    return new_state

# Shift operator
def shift_operator(state, links):
    new_state = [0] * num_sites
    for i in range(len(state)):
        if links[i]:
            new_state[(i + 1) % num_sites] += state[i]
            new_state[(i - 1) % num_sites] += state[i]
    return new_state

# Measure state
def measure_state(state):
    total_prob = sum(abs(amplitude) ** 2 for amplitude in state)
    rand = random.uniform(0, total_prob)
    cumulative = 0
    for i, amplitude in enumerate(state):
        cumulative += abs(amplitude) ** 2
        if cumulative > rand:
            return i
    return len(state) - 1

# Extract links from a webpage
def extract_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
        print(f"Retrieved content from {url}")  # Debugging line
        #Fetch Pause
        time.sleep(1)
        soup = BeautifulSoup(content, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')]
        if not links:
            print(f"No links found in the content of {url}")  # Debugging line
        else:
            print(f"Found links: {links}")  # Debugging line
        return links
    except requests.RequestException as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return []

# Main quantum walk algorithm
current_url = initial_url
crawled_urls = [current_url]

for step in range(1, num_steps + 1):
    print(f"Step {step}: {current_url}")
    
    links = extract_links(current_url)
    if not links:
        print(f"No links found at {current_url}")
        break
    
    link_exists = [1 if i < len(links) else 0 for i in range(num_sites)]
    
    # Apply quantum walk
    state = hadamard_coin(state)
    state = shift_operator(state, link_exists)
    
    # Measure the state to decide next URL
    next_index = measure_state(state)
    if next_index < len(links):
        current_url = links[next_index]
        crawled_urls.append(current_url)

# Write all crawled URLs to the file
with open(output_file, 'w') as f:
    for url in crawled_urls:
        f.write(f"{url}\n")

print(f"Final URL: {current_url}")
print(f"All crawled URLs have been saved to {output_file}")
