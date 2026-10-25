"""Retry transient Hub failures and verify a reopened stream's delivered prefix."""

import errno
import hashlib
import json
import logging
import time


def transient(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    # HTTP responses take precedence over a wrapped transport exception.
    for error in chain:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        status = status or getattr(error, "status_code", None) or getattr(error, "code", None)
        if isinstance(status, int):
            return status in (408, 425, 429) or 500 <= status < 600
    network_errnos = {errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET, errno.EHOSTUNREACH,
                      errno.ENETDOWN, errno.ENETRESET, errno.ENETUNREACH, errno.EPIPE, errno.ETIMEDOUT}
    for error in chain:
        if isinstance(error, (ConnectionError, TimeoutError)) or getattr(error, "errno", None) in network_errnos:
            return True
        for cls in type(error).__mro__:
            if cls.__module__.split(".")[0] in {"httpx", "httpcore", "httpx2", "httpcore2", "requests", "urllib3", "aiohttp", "http", "urllib", "socket"}:
                if cls.__name__ in {"TransportError", "NetworkError", "TimeoutException", "TimeoutError", "Timeout",
                                    "ConnectionError", "ProtocolError", "ChunkedEncodingError", "IncompleteRead",
                                    "RemoteDisconnected", "ClientConnectionError", "ClientPayloadError", "URLError", "gaierror"}:
                    return True
    return False


def retry(call, *, initial_delay=1, max_delay=60):
    delay = initial_delay
    while True:
        try:
            return call()
        except Exception as error:
            if not transient(error):
                raise
            logging.getLogger(__name__).warning("Transient Hub operation failure; retrying in %.1fs", delay)
            time.sleep(delay)
            delay = min(max_delay, 2*delay)


def replay(factory, *, fields, initial_delay=1, max_delay=60):
    """Resume remote reads without duplicating rows or accepting a changed prefix."""
    options = dict(initial_delay=initial_delay, max_delay=max_delay)
    delivered, prefix = 0, hashlib.sha256()
    end = object()

    def update(state, record):
        encoded = json.dumps({k: record[k] for k in fields if k in record}, sort_keys=True,
                             ensure_ascii=False, separators=(",", ":")).encode()
        state.update(len(encoded).to_bytes(8, "little"))
        state.update(encoded)

    def reopen():
        iterator, check = iter(factory()), hashlib.sha256()
        for _ in range(delivered):
            record = next(iterator, end)
            if record is end:
                raise ConnectionError("Reopened source ended before its delivered prefix")
            update(check, record)
        if check.digest() != prefix.digest():
            raise ValueError("Remote dataset changed its delivered prefix during replay")
        return iterator, next(iterator, end)

    iterator, record = retry(reopen, **options)
    while record is not end:
        update(prefix, record)
        delivered += 1
        yield record
        try:
            record = next(iterator, end)
        except Exception as error:
            if not transient(error):
                raise
        else:
            continue
        # Leave the transport exception context before checking replayed data.
        # A changed-prefix ValueError must not inherit a retryable cause.
        iterator, record = retry(reopen, **options)
