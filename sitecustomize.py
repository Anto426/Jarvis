import os
import socket


def _enabled(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


if _enabled(os.environ.get("JARVIS_FORCE_IPV4", "0")):
    _original_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo_ipv4
