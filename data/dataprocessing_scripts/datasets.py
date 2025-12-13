import pandas as pd
import random

# Load datasets

phish_df = pd.read_csv("../raw_data/verified_online.csv")
tranco = pd.read_csv("../raw_data/top-1m.csv", header=None, names=["rank", "domain"])

# Take top 100k domains
benign_domains = tranco.iloc[:100000]["domain"]
benign_urls = "https://" + benign_domains
benign_df = pd.DataFrame({"url": benign_urls, "label": 0})


patterns = [
    "https://{}",
    "https://www.{}/",
    "https://{}/home",
    "https://{}/login",
    "https://{}/account",
    "https://www.{}/products",
]

def generate_url(domain):
    pattern = random.choice(patterns)
    return pattern.format(domain)

benign_urls = [generate_url(d) for d in benign_domains]
benign_df = pd.DataFrame({"url": benign_urls, "label": 0})

# Select only URL column
phish_df = phish_df[["url"]].copy()

# Drop rows with missing or invalid URLs
phish_df = phish_df.dropna(subset=["url"])
phish_df = phish_df[phish_df["url"].str.startswith(("http", "https"))]

# Add label = 1 for phishing
phish_df["label"] = 1

print(len(phish_df))
phish_df["url"] = phish_df["url"].str.strip()
phish_df["url"] = phish_df["url"].str.replace(" ", "")
phish_df["url"] = phish_df["url"].astype(str)
phish_df["url"] = phish_df["url"].str.lower()


# Join datasets together
df = pd.concat([phish_df, benign_df], ignore_index=True)
df.to_csv("../basic_data.csv")

# Import all dependencies
import re
from urllib.parse import urlparse
from math import log2

# Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "update", "secure",
    "bank", "signin", "password", "confirm", "safe"
]

# ---- Helper functions ----

def is_ip(domain):
    """Check if domain is an IP address."""
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))

def entropy(s):
    """Calculate Shannon entropy of URL."""
    if len(s) == 0:
        return 0
    p = [s.count(c)/len(s) for c in set(s)]
    return -sum(px * log2(px) for px in p)

def extract_features(url):
    """Extract Level-1 features from a URL."""
    try:
        parsed = urlparse(url)
    except:
        return pd.Series([None] * 15)

    domain = parsed.netloc
    path = parsed.path

    # Basic numeric features
    url_length        = len(url)
    num_digits        = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    dot_count         = url.count(".")
    hyphen_in_domain  = 1 if "-" in domain else 0
    at_symbol         = 1 if "@" in url else 0
    double_slash      = 1 if url.count("//") > 1 else 0

    # Domain / Path features
    subdomain_count   = domain.count(".") - 1 if domain else 0
    domain_length     = len(domain)
    path_length       = len(path)

    # IP address check
    ip_flag = 1 if is_ip(domain) else 0

    # Suspicious keywords
    keyword_flag = 1 if any(k in url.lower() for k in SUSPICIOUS_KEYWORDS) else 0

    # Entropy
    url_entropy = entropy(url)

    return pd.Series([
        url_length,
        num_digits,
        num_special_chars,
        dot_count,
        hyphen_in_domain,
        at_symbol,
        double_slash,
        subdomain_count,
        domain_length,
        path_length,
        ip_flag,
        keyword_flag,
        url_entropy
    ])

# ---- Apply to dataset ----

TIER1_COLS = [
    "url_length",
    "num_digits",
    "num_special_chars",
    "dot_count",
    "hyphen_in_domain",
    "at_symbol",
    "double_slash",
    "subdomain_count",
    "domain_length",
    "path_length",
    "ip_flag",
    "keyword_flag",
    "url_entropy"
]

df[TIER1_COLS] = df["url"].apply(extract_features)

df.to_csv("../tier1_data.csv")

# 11. Subdomain count
def count_subdomains(domain):
    if not domain:
        return 0
    parts = domain.split(".")
    # Remove TLD + main domain → subdomains left
    return max(0, len(parts) - 2)

# 13. Domain length + TLD flag
# ---------------------------
malicious_tlds = {".ru", ".tk", ".ml", ".xyz", ".info", ".top", ".ga", ".gq", ".cf"}

def domain_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc

    if not domain:
        return pd.Series([0, 0])

    domain_len = len(domain)

    # Extract TLD (.com, .net, .ru, etc.)
    tld = "." + domain.split(".")[-1]

    tld_flag = 1 if tld in malicious_tlds else 0

    return pd.Series([domain_len, tld_flag])

# 14. Encoded characters flag
def encoded_char_flag(url):
    encodings = ["%20", "%2F", "%3D", "%3F", "%40", "%25"]
    return 1 if any(code in url for code in encodings) else 0


# 15. Path length
def path_length(url):
    parsed = urlparse(url)
    return len(parsed.path) if parsed.path else 0

# ---------------------------
# Apply Tier-2 features
df["url_entropy"] = df["url"].apply(entropy)

df["subdomain_count"] = df["url"].apply(
    lambda x: count_subdomains(urlparse(x).netloc)
)
df[["domain_length_t2", "malicious_tld_flag"]] = df["url"].apply(domain_features)

df["encoded_flag"] = df["url"].apply(encoded_char_flag)

df["path_length_t2"] = df["url"].apply(path_length)


df.to_csv("../tier2_data.csv")
