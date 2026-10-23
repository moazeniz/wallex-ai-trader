"""
Dune Analytics API Client

Provides access to on-chain analytics via SQL queries:
- Execute custom queries
- Get query results
- Access pre-built dashboards

Rate Limits (Free Tier):
- 2,500 credits/month
- 30-minute query timeout
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import get_config

logger = logging.getLogger(__name__)


class DuneClient:
    """
    Client for Dune Analytics API.

    Allows executing SQL queries on blockchain data
    and retrieving results from existing queries.
    """

    BASE_URL = "https://api.dune.com/api/v1"

    # Pre-built query IDs for common analytics
    # These are example public query IDs - replace with your own
    QUERIES = {
        "eth_whale_transactions": 3234567,  # Example - replace with real query
        "defi_tvl": 3234568,
        "dex_volume": 3234569,
        "nft_activity": 3234570,
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Dune client.

        Args:
            api_key: API key (uses config if not provided)
        """
        config = get_config()
        self.api_key = api_key or config.market_data.dune_api_key
        self.rate_limit = config.market_data.dune_rate_limit
        self._last_request = 0

        if not self.api_key:
            logger.warning("Dune API key not configured")

        self.headers = {
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        if not self.api_key:
            raise ValueError("Dune API key not configured")

        self._rate_limit_wait()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=60)
            else:
                response = requests.post(url, headers=self.headers, json=data, timeout=60)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Dune request failed: {e}")
            raise

    def execute_query(
        self,
        query_id: int,
        parameters: Optional[Dict] = None,
    ) -> str:
        """
        Execute a Dune query.

        Args:
            query_id: Dune query ID
            parameters: Query parameters

        Returns:
            Execution ID
        """
        data = {}
        if parameters:
            data["query_parameters"] = parameters

        result = self._request("POST", f"query/{query_id}/execute", data=data)
        return result.get("execution_id")

    def get_execution_status(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Get status of a query execution.

        Args:
            execution_id: Execution ID

        Returns:
            Status information
        """
        result = self._request("GET", f"execution/{execution_id}/status")
        return result

    def get_execution_results(
        self,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Get results of a completed query.

        Args:
            execution_id: Execution ID

        Returns:
            Query results
        """
        result = self._request("GET", f"execution/{execution_id}/results")
        return result

    def execute_and_wait(
        self,
        query_id: int,
        parameters: Optional[Dict] = None,
        timeout: int = 300,  # 5 minutes
        poll_interval: int = 5,
    ) -> List[Dict]:
        """
        Execute a query and wait for results.

        Args:
            query_id: Dune query ID
            parameters: Query parameters
            timeout: Max wait time in seconds
            poll_interval: Seconds between status checks

        Returns:
            Query result rows
        """
        execution_id = self.execute_query(query_id, parameters)

        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_execution_status(execution_id)
            state = status.get("state")

            if state == "QUERY_STATE_COMPLETED":
                results = self.get_execution_results(execution_id)
                return results.get("result", {}).get("rows", [])
            elif state == "QUERY_STATE_FAILED":
                raise Exception(f"Query failed: {status.get('error')}")
            elif state == "QUERY_STATE_CANCELLED":
                raise Exception("Query was cancelled")

            time.sleep(poll_interval)

        raise TimeoutError(f"Query did not complete within {timeout} seconds")

    def get_latest_results(
        self,
        query_id: int,
    ) -> List[Dict]:
        """
        Get latest results from a query (uses cached results).

        Args:
            query_id: Dune query ID

        Returns:
            Query result rows
        """
        result = self._request("GET", f"query/{query_id}/results")
        return result.get("result", {}).get("rows", [])

    def get_whale_transactions(
        self,
        token_address: str,
        min_value_usd: float = 100000,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get large token transactions (whale activity).

        Note: This requires a custom query to be set up in Dune.

        Args:
            token_address: Token contract address
            min_value_usd: Minimum transaction value
            limit: Max results

        Returns:
            List of whale transactions
        """
        # Example - you would need to create this query in Dune
        # and update the query ID
        query_id = self.QUERIES.get("eth_whale_transactions")
        if not query_id:
            logger.warning("Whale transactions query not configured")
            return []

        try:
            return self.execute_and_wait(
                query_id=query_id,
                parameters={
                    "token_address": token_address,
                    "min_value": min_value_usd,
                    "limit": limit,
                },
            )
        except Exception as e:
            logger.error(f"Failed to get whale transactions: {e}")
            return []

    def get_dex_volume(
        self,
        token_address: Optional[str] = None,
        days: int = 7,
    ) -> List[Dict]:
        """
        Get DEX trading volume.

        Args:
            token_address: Optional token filter
            days: Number of days

        Returns:
            Volume data
        """
        query_id = self.QUERIES.get("dex_volume")
        if not query_id:
            return []

        try:
            params = {"days": days}
            if token_address:
                params["token_address"] = token_address

            return self.get_latest_results(query_id)
        except Exception as e:
            logger.error(f"Failed to get DEX volume: {e}")
            return []

    def run_custom_query(
        self,
        sql: str,
        timeout: int = 300,
    ) -> List[Dict]:
        """
        Run a custom SQL query.

        Note: Requires appropriate Dune plan for query creation.

        Args:
            sql: SQL query
            timeout: Max wait time

        Returns:
            Query results
        """
        # This would require creating a query first via the API
        # For now, log a warning
        logger.warning("Custom SQL execution not yet implemented - use Dune UI")
        return []


def create_dune_client(
    api_key: Optional[str] = None,
) -> DuneClient:
    """Create a DuneClient instance."""
    return DuneClient(api_key=api_key)


if __name__ == "__main__":
    print("Testing Dune Analytics Client...")

    client = DuneClient()

    if not client.api_key:
        print("Error: DUNE_API_KEY not set")
        exit(1)

    print("\nNote: Dune requires setting up custom queries in their UI.")
    print("The query IDs in this client are examples only.")

    # Example: Try to get results from a public query
    print("\n1. Attempting to fetch from example query...")
    try:
        # This will likely fail without valid query ID
        results = client.get_latest_results(3234567)
        print(f"  Got {len(results)} rows")
    except Exception as e:
        print(f"  Expected error (no valid query ID): {e}")

    print("\nTo use Dune effectively:")
    print("1. Create queries in Dune UI (dune.com)")
    print("2. Note the query IDs")
    print("3. Update QUERIES dict in this client")
    print("4. Use get_latest_results() or execute_and_wait()")

    print("\nTest complete!")
