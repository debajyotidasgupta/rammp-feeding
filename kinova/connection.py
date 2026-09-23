"""Connection helpers for a Kinova Gen3 arm over the Kortex API (2.x, TCP transport).

The Kortex API exposes two channels:

  * TCP  :10000 -- request/response ("high level"): move, read state, config.
  * UDP  :10001 -- real-time cyclic ("low level"): 1 kHz torque/velocity control.

`KinovaConnection` owns the transport, router and session lifecycle for both, so
callers just ask for the client stubs they need and get cleanup for free.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.messages import Session_pb2

DEFAULT_IP = os.environ.get("KINOVA_IP", "192.168.2.67")
DEFAULT_USERNAME = os.environ.get("KINOVA_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("KINOVA_PASSWORD", "admin")

TCP_PORT = 10000
UDP_PORT = 10001


@dataclass
class ConnectionConfig:
    ip: str = DEFAULT_IP
    username: str = DEFAULT_USERNAME
    password: str = DEFAULT_PASSWORD
    # The arm drops the session if it hears nothing for this long (ms).
    session_inactivity_timeout: int = 60_000
    connection_inactivity_timeout: int = 2_000
    # Applied to blocking RPCs; the arm is on a LAN so this is generous.
    rpc_timeout_s: int = 10


class KinovaConnection:
    """Context manager owning transports, routers and sessions for one arm.

    Usage::

        with KinovaConnection() as conn:
            print(conn.base.GetArmState())

    `real_time=True` additionally opens the UDP cyclic channel and exposes
    `conn.base_cyclic_rt`. Only open it when you actually need 1 kHz feedback.
    """

    def __init__(self, config: ConnectionConfig | None = None, real_time: bool = False):
        self.config = config or ConnectionConfig()
        self.real_time = real_time

        self._tcp_transport: TCPTransport | None = None
        self._tcp_router: RouterClient | None = None
        self._tcp_session: SessionManager | None = None

        self._udp_transport: UDPTransport | None = None
        self._udp_router: RouterClient | None = None
        self._udp_session: SessionManager | None = None

        self.base: BaseClient | None = None
        self.base_cyclic: BaseCyclicClient | None = None
        self.base_cyclic_rt: BaseCyclicClient | None = None
        self.device_config: DeviceConfigClient | None = None

        self.last_error: Exception | None = None

        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = self.config.rpc_timeout_s * 1000

    # -- session plumbing -------------------------------------------------

    def _session_info(self) -> Session_pb2.CreateSessionInfo:
        info = Session_pb2.CreateSessionInfo()
        info.username = self.config.username
        info.password = self.config.password
        info.session_inactivity_timeout = self.config.session_inactivity_timeout
        info.connection_inactivity_timeout = self.config.connection_inactivity_timeout
        return info

    def _on_error(self, ex) -> None:
        """Router error callback.

        This runs on the transport's *receive* thread, inside the frame loop --
        raising here would tear that thread down and the connection would go
        quiet instead of reporting. So record and log, never raise.
        """
        self.last_error = ex
        print(f"[kinova] router error: {ex}", file=sys.stderr)

    def connect(self) -> "KinovaConnection":
        cfg = self.config

        self._tcp_transport = TCPTransport()
        self._tcp_transport.connect(cfg.ip, TCP_PORT)
        self._tcp_router = RouterClient(self._tcp_transport, self._on_error)
        self._tcp_session = SessionManager(self._tcp_router)
        self._tcp_session.CreateSession(self._session_info())

        self.base = BaseClient(self._tcp_router)
        self.base_cyclic = BaseCyclicClient(self._tcp_router)
        self.device_config = DeviceConfigClient(self._tcp_router)

        if self.real_time:
            self._udp_transport = UDPTransport()
            self._udp_transport.connect(cfg.ip, UDP_PORT)
            self._udp_router = RouterClient(self._udp_transport, self._on_error)
            self._udp_session = SessionManager(self._udp_router)
            self._udp_session.CreateSession(self._session_info())
            self.base_cyclic_rt = BaseCyclicClient(self._udp_router)

        return self

    def close(self) -> None:
        # Close sessions before transports, real-time first; ignore teardown
        # errors so one bad channel cannot mask the original exception.
        for session in (self._udp_session, self._tcp_session):
            if session is not None:
                try:
                    session.CloseSession()
                except Exception:
                    pass
        for transport in (self._udp_transport, self._tcp_transport):
            if transport is not None:
                try:
                    transport.disconnect()
                except Exception:
                    pass
        self._udp_session = self._tcp_session = None
        self._udp_transport = self._tcp_transport = None

    def __enter__(self) -> "KinovaConnection":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


@contextmanager
def kinova_connection(ip: str | None = None, real_time: bool = False, **kwargs):
    """Convenience wrapper: `with kinova_connection("192.168.2.67") as conn: ...`"""
    config = ConnectionConfig(**({"ip": ip} if ip else {}), **kwargs)
    conn = KinovaConnection(config, real_time=real_time)
    try:
        yield conn.connect()
    finally:
        conn.close()
