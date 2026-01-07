"""
WebSocket-based streaming for simulation visualization.

Provides real-time streaming of rendered frames to a browser viewer.
"""

from .server import StreamServer, StreamConfig
from .client import create_viewer_html, save_viewer_html
from .streaming_sim import StreamingSimulator

__all__ = [
    "StreamServer",
    "StreamConfig",
    "StreamingSimulator",
    "create_viewer_html",
    "save_viewer_html",
]
