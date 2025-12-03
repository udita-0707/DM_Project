"""
Semantic Scholar API Integration

This module provides citation data functionality using the Semantic Scholar API.
Includes caching, rate limiting, and batch processing capabilities.

Usage:
    from src.data_acquisition.semanticscholar_api import SemanticScholarAPI
    
    api = SemanticScholarAPI(api_key="your_key_here")
    citation_count = api.get_citation_count("2101.12345")
"""

import requests
import json
import time
import os
from typing import Optional, Dict, List
from datetime import datetime

class SemanticScholarAPI:
    """
    Enhanced Semantic Scholar API client with caching and rate limiting.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RATE_LIMIT_PER_SECOND = 1  # 1 request per second
    DEFAULT_FIELDS = "citationCount,referenceCount,title,year,authors"
    
    def __init__(self, api_key: Optional[str] = None, cache_file: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: Semantic Scholar API key (optional but recommended)
            cache_file: Path to JSON file for caching responses (optional)
        """
        self.api_key = api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self.headers = {}
        if self.api_key:
            self.headers['x-api-key'] = self.api_key
        
        # Load cache if provided
        self.cache = {}
        self.cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached entries from {cache_file}")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        
        self.last_request_time = 0
        self.request_count = 0
        self.total_requests = 0
        self._cache_save_counter = 0  # Counter to batch cache saves
    
    def _save_cache(self, force: bool = False):
        """
        Save cache to file if cache_file is set.
        
        Args:
            force: If True, save immediately. Otherwise, save every 50 entries.
        """
        if self.cache_file:
            self._cache_save_counter += 1
            # Save every 50 entries or if forced
            if force or self._cache_save_counter >= 50:
                try:
                    with open(self.cache_file, 'w') as f:
                        json.dump(self.cache, f, indent=2)
                    self._cache_save_counter = 0
                except Exception as e:
                    print(f"Warning: Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting (1 request per second)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.RATE_LIMIT_PER_SECOND:
            sleep_time = self.RATE_LIMIT_PER_SECOND - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None, data: Dict = None, 
                     method: str = 'GET', max_retries: int = 3) -> Optional[Dict]:
        """
        Make an API request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters (for GET requests)
            data: Request body data (for POST requests)
            method: HTTP method ('GET' or 'POST')
            max_retries: Maximum number of retry attempts
        
        Returns:
            JSON response data or None if failed
        """
        for attempt in range(max_retries):
            self._rate_limit()
            
            try:
                if method.upper() == 'POST':
                    response = requests.post(url, json=data, params=params, 
                                            headers=self.headers, timeout=30)
                else:
                    response = requests.get(url, params=params, headers=self.headers, timeout=10)
                
                self.total_requests += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit exceeded - exponential backoff
                    wait_time = (2 ** attempt) * 60  # 60s, 120s, 240s
                    print(f"Rate limit exceeded. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 400:
                    # Bad request (e.g., too many IDs, invalid format) - don't retry
                    print(f"Bad request (400): {response.text[:200]}")
                    return None
                elif response.status_code == 404:
                    # Paper not found
                    return None
                else:
                    # Other errors - only retry for 5xx server errors
                    print(f"API error {response.status_code}: {response.text[:200]}")
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        # Server error - retry with exponential backoff
                        time.sleep(2 ** attempt)
                        continue
                    # Client errors (4xx) - don't retry
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    def get_paper_data(self, arxiv_id: str, fields: Optional[str] = None) -> Optional[Dict]:
        """
        Get paper data from Semantic Scholar using arXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2101.12345")
            fields: Comma-separated list of fields to retrieve (default: citationCount,referenceCount,title,year,authors)
        
        Returns:
            Dictionary with paper data or None if not found
        """
        # Check cache first
        cache_key = f"ARXIV:{arxiv_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Make API request
        url = f"{self.BASE_URL}/paper/ARXIV:{arxiv_id}"
        params = {"fields": fields or self.DEFAULT_FIELDS}
        
        data = self._make_request(url, params, method='GET')
        
        # Cache the result (even if None to avoid repeated failed lookups)
        self.cache[cache_key] = data
        
        return data
    
    def get_citation_count(self, arxiv_id: str) -> Optional[int]:
        """
        Get citation count for a paper using its arXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2101.12345")
        
        Returns:
            Citation count (int) or None if not found
        """
        data = self.get_paper_data(arxiv_id, fields="citationCount")
        if data:
            return data.get('citationCount', 0)
        return None
    
    def batch_get_papers(self, arxiv_ids: List[str], fields: Optional[str] = None, 
                        batch_size: int = 500) -> Dict[str, Optional[Dict]]:
        """
        Get multiple papers in batch requests (more efficient for large datasets).
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            fields: Comma-separated list of fields to retrieve
            batch_size: Number of papers per batch (max 500 per API limit)
        
        Returns:
            Dictionary mapping arxiv_id -> paper_data or None
        """
        results = {}
        fields = fields or self.DEFAULT_FIELDS
        
        # Filter out cached entries first
        # Only use cache if value is not None (None means previous fetch failed, retry it)
        uncached_ids = []
        for arxiv_id in arxiv_ids:
            cache_key = f"ARXIV:{arxiv_id}"
            if cache_key in self.cache and self.cache[cache_key] is not None:
                results[arxiv_id] = self.cache[cache_key]
            else:
                uncached_ids.append(arxiv_id)
        
        if not uncached_ids:
            return results
        
        # Split into batches (API limit is 500 per request)
        batch_size = min(batch_size, 500)
        total_batches = (len(uncached_ids) + batch_size - 1) // batch_size
        
        print(f"Fetching {len(uncached_ids)} uncached papers in {total_batches} batch(es)...")
        print(f"  (Skipping {len(arxiv_ids) - len(uncached_ids)} papers already in cache)")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(uncached_ids))
            batch_ids = uncached_ids[start_idx:end_idx]
            
            # Format IDs for API (ARXIV: prefix)
            api_ids = [f"ARXIV:{arxiv_id}" for arxiv_id in batch_ids]
            
            # Make batch request
            url = f"{self.BASE_URL}/paper/batch"
            params = {"fields": fields}
            data = {"ids": api_ids}
            
            print(f"  Making batch API request for {len(batch_ids)} papers...")
            batch_response = self._make_request(url, params=params, data=data, method='POST')
            print(f"  API request completed. Response type: {type(batch_response)}")
            
            # Rate limit: wait 1 second between batches (1 request per second limit)
            if batch_num < total_batches - 1:  # Don't wait after last batch
                time.sleep(1)
            
            if batch_response:
                if isinstance(batch_response, list):
                    # API returns papers as a list in the same order as requested IDs
                for i, paper_data in enumerate(batch_response):
                    if i < len(batch_ids):
                        arxiv_id = batch_ids[i]  # Papers are returned in request order
                        
                        if paper_data and isinstance(paper_data, dict):
                            results[arxiv_id] = paper_data
                            self.cache[f"ARXIV:{arxiv_id}"] = paper_data
                        else:
                            # Paper not found or invalid response
                            results[arxiv_id] = None
                            self.cache[f"ARXIV:{arxiv_id}"] = None
                
                # Mark any missing papers as None (in case response is shorter)
                for arxiv_id in batch_ids:
                    if arxiv_id not in results:
                            results[arxiv_id] = None
                            self.cache[f"ARXIV:{arxiv_id}"] = None
                elif isinstance(batch_response, dict):
                    # API returns papers as a dictionary with IDs as keys
                    for arxiv_id in batch_ids:
                        api_id = f"ARXIV:{arxiv_id}"
                        paper_data = batch_response.get(api_id)
                        
                        if paper_data and isinstance(paper_data, dict):
                            results[arxiv_id] = paper_data
                            self.cache[f"ARXIV:{arxiv_id}"] = paper_data
                        else:
                            results[arxiv_id] = None
                            self.cache[f"ARXIV:{arxiv_id}"] = None
                else:
                    print(f"  Warning: Unexpected batch response type: {type(batch_response)}")
                    # Mark all as None
                    for arxiv_id in batch_ids:
                        results[arxiv_id] = None
                        self.cache[f"ARXIV:{arxiv_id}"] = None
            else:
                # Batch request failed (None response)
                print(f"  Warning: Batch request failed or returned None")
                for arxiv_id in batch_ids:
                    results[arxiv_id] = None
                    self.cache[f"ARXIV:{arxiv_id}"] = None
            
            print(f"  Batch {batch_num + 1}/{total_batches} complete ({len(batch_ids)} papers)")
        
        return results
    
    def batch_get_citations(self, arxiv_ids: List[str], progress_callback: Optional[callable] = None,
                           use_batch_api: bool = False) -> Dict[str, Optional[int]]:
        """
        Get citation counts for multiple papers with progress tracking.
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            progress_callback: Optional callback function(current, total) for progress updates
            use_batch_api: If True, use batch API endpoint (faster for large datasets)
        
        Returns:
            Dictionary mapping arxiv_id -> citation_count
        """
        if use_batch_api and len(arxiv_ids) > 10:
            # Use batch API for efficiency
            papers_data = self.batch_get_papers(arxiv_ids, fields="citationCount")
            results = {}
            for arxiv_id, paper_data in papers_data.items():
                if paper_data:
                    results[arxiv_id] = paper_data.get('citationCount', 0)
                else:
                    results[arxiv_id] = None
            return results
        
        # Fallback to individual requests
        results = {}
        total = len(arxiv_ids)
        
        print(f"Fetching citation data for {total} papers...")
        print(f"Estimated time: ~{total / 3600:.1f} hours at 1 req/sec")
        
        for i, arxiv_id in enumerate(arxiv_ids, 1):
            citation_count = self.get_citation_count(arxiv_id)
            results[arxiv_id] = citation_count
            
            if progress_callback:
                progress_callback(i, total)
            elif i % 100 == 0 or i == total:
                print(f"Progress: {i}/{total} ({100*i/total:.1f}%) - Total API calls: {self.total_requests}")
        
        return results
    
    def save_cache(self):
        """Force save cache to disk."""
        self._save_cache(force=True)
    
    def get_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            'total_requests': self.total_requests,
            'cached_entries': len(self.cache),
            'api_key_configured': self.api_key is not None
        }


# Backward compatibility function
def get_citation_count(arxiv_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """
    Legacy function for backward compatibility.
    
    Args:
        arxiv_id: ArXiv paper ID
        api_key: Optional API key (or set SEMANTIC_SCHOLAR_API_KEY env var)
    
    Returns:
        Citation count or None
    """
    api = SemanticScholarAPI(api_key=api_key)
    return api.get_citation_count(arxiv_id)


if __name__ == "__main__":
    print("Semantic Scholar API module loaded.")
    print("\nUsage:")
    print("  from src.data_acquisition.semanticscholar_api import SemanticScholarAPI")
    print("  api = SemanticScholarAPI(api_key='your_key')")
    print("  citation = api.get_citation_count('2101.12345')")
    print("\nOr set environment variable:")
    print("  export SEMANTIC_SCHOLAR_API_KEY='your_key'")
