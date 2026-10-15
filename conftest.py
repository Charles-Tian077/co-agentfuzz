"""Deny network/model connections during tests, including loopback HTTP.

Only Windows asyncio's internal socketpair handshake is exempt.
"""
import inspect
import os
import socket
from pathlib import Path

# CLI subprocesses use the same source checkout, without an editable install.
_source_dir = str(Path(__file__).resolve().parent / 'src')
os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, (
    _source_dir, os.environ.get('PYTHONPATH', ''))))

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _asyncio_socketpair():
    frames = inspect.stack(context=0)
    legacy_pair = any(frame.function == 'socketpair' and
                      frame.filename.replace('\\', '/').endswith('/asyncio/windows_utils.py')
                      for frame in frames)
    stdlib_pair = any(frame.function == '_fallback_socketpair' and
                      frame.filename.replace('\\', '/').endswith('/socket.py')
                      for frame in frames)
    asyncio_pipe = any(frame.function == '_make_self_pipe' and
                       frame.filename.replace('\\', '/').endswith('/asyncio/proactor_events.py')
                       for frame in frames)
    return legacy_pair or (stdlib_pair and asyncio_pipe)


def _offline_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not _asyncio_socketpair():
        raise RuntimeError('Network and real model calls are disabled in the unit test suite')
    return _original_connect(sock, address)


def _offline_connect_ex(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not _asyncio_socketpair():
        raise RuntimeError('Network and real model calls are disabled in the unit test suite')
    return _original_connect_ex(sock, address)


def pytest_sessionstart(session):
    socket.socket.connect = _offline_connect
    socket.socket.connect_ex = _offline_connect_ex


def pytest_sessionfinish(session, exitstatus):
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
