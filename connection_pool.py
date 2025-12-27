"""
Connection Pool for LLM Requests
Provides connection pooling, retry logic, and better performance for LLM backends
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from aiohttp import ClientTimeout, ClientError
import backoff

logger = logging.getLogger(__name__)

@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    avg_response_time: float = 0
    last_request_time: Optional[datetime] = None
    pool_hits: int = 0
    pool_misses: int = 0


class ConnectionPool:
    """
    Manages HTTP connection pooling for LLM backends with retry logic

    Features:
    - Connection reuse for better performance
    - Exponential backoff retry logic
    - Connection health monitoring
    - Request timeout handling
    - Statistics tracking
    """

    def __init__(self,
                 max_connections: int = 10,
                 max_keepalive_connections: int = 5,
                 keepalive_timeout: int = 30,
                 default_timeout: int = 180,
                 max_retries: int = 3):
        """
        Initialize connection pool

        Args:
            max_connections: Maximum total connections
            max_keepalive_connections: Max persistent connections
            keepalive_timeout: Keepalive timeout in seconds
            default_timeout: Default request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_timeout = keepalive_timeout
        self.default_timeout = default_timeout
        self.max_retries = max_retries

        # Connection pools per endpoint
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._stats: Dict[str, ConnectionStats] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        # Global connector for shared connections
        self._connector = None
        self._closed = False

    async def _get_connector(self) -> aiohttp.TCPConnector:
        """Get or create the global connector"""
        # Check if connector exists, is not closed, and is in the correct event loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = asyncio.get_event_loop()

        if (self._connector is None or
            self._connector.closed or
            self._connector._loop != current_loop):
            # Close old connector if it exists
            if self._connector and not self._connector.closed:
                await self._connector.close()
            # Create new connector in current event loop (don't pass loop parameter - deprecated)
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_keepalive_connections,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                keepalive_timeout=self.keepalive_timeout
            )
        return self._connector

    async def _get_session(self, endpoint: str) -> aiohttp.ClientSession:
        """
        Get or create a session for an endpoint

        Args:
            endpoint: The base URL endpoint

        Returns:
            aiohttp.ClientSession for the endpoint
        """
        # Check if we need to recreate sessions due to event loop change
        try:
            current_loop = asyncio.get_running_loop()
            # If connector loop changed, close all sessions
            if (self._connector and hasattr(self._connector, '_loop') and
                self._connector._loop != current_loop):
                await self._close_all_sessions()
        except RuntimeError:
            pass

        if endpoint not in self._sessions or self._sessions[endpoint].closed:
            if endpoint not in self._locks:
                self._locks[endpoint] = asyncio.Lock()

            # Add timeout to lock acquisition to prevent deadlock
            try:
                logger.debug(f"Attempting to acquire lock for {endpoint}")

                # Use wait_for with lock acquisition for Python 3.10 compatibility
                async def acquire_and_create():
                    async with self._locks[endpoint]:
                        logger.debug(f"Lock acquired for {endpoint}")
                        # Double-check after acquiring lock
                        if endpoint not in self._sessions or self._sessions[endpoint].closed:
                            connector = await self._get_connector()

                            # Disable timeout at session level - we handle it with asyncio.wait_for
                            self._sessions[endpoint] = aiohttp.ClientSession(
                                connector=connector,
                                headers={'User-Agent': 'WhimWeaver/1.0'},
                                timeout=aiohttp.ClientTimeout(total=None)  # Disable all timeouts
                            )

                            if endpoint not in self._stats:
                                self._stats[endpoint] = ConnectionStats()

                            self._stats[endpoint].pool_misses += 1
                            logger.debug(f"Created new session for {endpoint}")

                await asyncio.wait_for(acquire_and_create(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error(f"❌ DEADLOCK DETECTED: Lock acquisition timeout for {endpoint} after 5 seconds")
                logger.error(f"   Resetting lock to recover from deadlock")
                # Reset the stuck lock to recover
                self._locks[endpoint] = asyncio.Lock()
                # Retry once with the new lock
                async with self._locks[endpoint]:
                    logger.info(f"✅ Lock reset successful, retrying session creation for {endpoint}")
                    if endpoint not in self._sessions or self._sessions[endpoint].closed:
                        connector = await self._get_connector()
                        self._sessions[endpoint] = aiohttp.ClientSession(
                            connector=connector,
                            headers={'User-Agent': 'WhimWeaver/1.0'},
                            timeout=aiohttp.ClientTimeout(total=None)
                        )
                        if endpoint not in self._stats:
                            self._stats[endpoint] = ConnectionStats()
                        self._stats[endpoint].pool_misses += 1
                        logger.info(f"Session created after lock reset for {endpoint}")
        else:
            self._stats[endpoint].pool_hits += 1

        return self._sessions[endpoint]

    async def _close_all_sessions(self):
        """Close all existing sessions"""
        for session in self._sessions.values():
            if not session.closed:
                await session.close()
        self._sessions.clear()
        logger.debug("Closed all sessions due to event loop change")

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def request(self,
                      method: str,
                      url: str,
                      **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic

        Args:
            method: HTTP method (GET, POST, etc)
            url: Full URL for the request
            **kwargs: Additional arguments for the request

        Returns:
            Response data as dict

        Raises:
            aiohttp.ClientError: If request fails after retries
        """
        # Extract endpoint from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        endpoint = f"{parsed.scheme}://{parsed.netloc}"

        # Get or create session
        session = await self._get_session(endpoint)

        # Update stats
        if endpoint not in self._stats:
            self._stats[endpoint] = ConnectionStats()
        stats = self._stats[endpoint]
        stats.total_requests += 1

        start_time = datetime.now()

        try:
            # Remove timeout from kwargs to avoid "used inside a task" error
            # We'll use asyncio.wait_for instead
            timeout_value = kwargs.pop('timeout', self.default_timeout)
            if isinstance(timeout_value, aiohttp.ClientTimeout):
                timeout_value = timeout_value.total

            async def make_request():
                # Session already has timeout=None, so no need to pass it again
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()

            # Use asyncio.wait_for instead of aiohttp timeout
            data = await asyncio.wait_for(make_request(), timeout=timeout_value)

            # Update success stats
            stats.successful_requests += 1
            elapsed = (datetime.now() - start_time).total_seconds()

            # Update average response time
            if stats.avg_response_time == 0:
                stats.avg_response_time = elapsed
            else:
                # Moving average
                stats.avg_response_time = (stats.avg_response_time * 0.9) + (elapsed * 0.1)

            stats.last_request_time = datetime.now()

            logger.debug(f"Request to {url} completed in {elapsed:.2f}s")
            return data

        except Exception as e:
            stats.failed_requests += 1
            logger.error(f"Request to {url} failed: {e}")
            raise

    async def post(self, url: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Convenience method for POST requests

        Args:
            url: Full URL for the request
            json: JSON data to send
            **kwargs: Additional arguments

        Returns:
            Response data as dict
        """
        return await self.request('POST', url, json=json, **kwargs)

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Convenience method for GET requests

        Args:
            url: Full URL for the request
            **kwargs: Additional arguments

        Returns:
            Response data as dict
        """
        return await self.request('GET', url, **kwargs)

    async def health_check(self, endpoint: str, path: str = '/health') -> bool:
        """
        Check if an endpoint is healthy

        Args:
            endpoint: Base endpoint URL
            path: Health check path

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{endpoint.rstrip('/')}{path}"
            # Use shorter timeout for health checks
            await self.get(url, timeout=ClientTimeout(total=5))
            return True
        except Exception as e:
            logger.debug(f"Health check failed for {endpoint}: {e}")
            return False

    def get_stats(self, endpoint: Optional[str] = None) -> Union[ConnectionStats, Dict[str, ConnectionStats]]:
        """
        Get connection statistics

        Args:
            endpoint: Specific endpoint or None for all

        Returns:
            Stats for endpoint or all endpoints
        """
        if endpoint:
            return self._stats.get(endpoint, ConnectionStats())
        return dict(self._stats)

    async def close(self):
        """Close all connections and clean up"""
        if self._closed:
            return

        self._closed = True

        # Close all sessions
        for session in self._sessions.values():
            if not session.closed:
                await session.close()

        # Close connector
        if self._connector and not self._connector.closed:
            await self._connector.close()

        self._sessions.clear()
        logger.info("Connection pool closed")

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


# Global connection pool instance
_global_pool: Optional[ConnectionPool] = None


def get_connection_pool(**kwargs) -> ConnectionPool:
    """
    Get or create the global connection pool

    Args:
        **kwargs: Arguments for ConnectionPool if creating new

    Returns:
        Global ConnectionPool instance
    """
    global _global_pool

    if _global_pool is None:
        _global_pool = ConnectionPool(**kwargs)

    return _global_pool


async def close_global_pool():
    """Close the global connection pool"""
    global _global_pool

    if _global_pool:
        await _global_pool.close()
        _global_pool = None