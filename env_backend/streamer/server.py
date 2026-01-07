"""
WebSocket server for streaming simulation frames.
"""

import asyncio
import json
import threading
import time
from typing import Optional, Callable, Dict, Any, Set
from dataclasses import dataclass, field
import numpy as np
from io import BytesIO

try:
    import websockets
    from websockets.server import serve as websocket_serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class StreamConfig:
    """Configuration for the stream server."""
    host: str = "localhost"
    port: int = 8765
    
    # Frame settings
    max_fps: float = 30.0
    jpeg_quality: int = 80
    
    # Connection settings
    max_clients: int = 10


class StreamServer:
    """
    WebSocket server for streaming simulation frames to browsers.
    
    Runs in a separate thread and accepts frame updates from the
    simulation. Connected clients receive JPEG-compressed frames.
    
    Usage:
        server = StreamServer(port=8765)
        server.start()
        
        # In simulation loop:
        frame = renderer.render(data)
        server.send_frame(frame)
        
        # When done:
        server.stop()
    """
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize the stream server.
        
        Args:
            config: Stream configuration
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required for streaming. "
                "Install with: pip install websockets"
            )
        if not HAS_PIL:
            raise ImportError(
                "PIL/Pillow required for frame compression. "
                "Install with: pip install Pillow"
            )
        
        self.config = config or StreamConfig()
        
        # Server state
        self._clients: Set = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Frame state
        self._current_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_time = 0.0
        self._min_frame_interval = 1.0 / self.config.max_fps
        
        # Stats
        self._frames_sent = 0
        self._bytes_sent = 0
        self._start_time = 0.0
    
    def start(self):
        """Start the server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        print(f"Stream server started at ws://{self.config.host}:{self.config.port}")
        print(f"Open the viewer HTML in a browser to see the stream")
    
    def stop(self):
        """Stop the server."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        print("Stream server stopped")
    
    def send_frame(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Send a frame to all connected clients.
        
        Args:
            frame: RGB image as (H, W, 3) uint8 array
            metadata: Optional metadata to send with frame
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self._frame_time < self._min_frame_interval:
            return
        self._frame_time = current_time
        
        # Compress frame to JPEG
        jpeg_data = self._compress_frame(frame)
        
        with self._frame_lock:
            self._current_frame = jpeg_data
        
        # Send to clients asynchronously
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_frame(jpeg_data, metadata),
                self._loop
            )
    
    def _compress_frame(self, frame: np.ndarray) -> bytes:
        """Compress frame to JPEG bytes."""
        img = Image.fromarray(frame)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
        return buffer.getvalue()
    
    async def _broadcast_frame(self, jpeg_data: bytes, metadata: Optional[Dict] = None):
        """Broadcast frame to all clients."""
        if not self._clients:
            return
        
        # Create message with optional metadata
        import base64
        frame_b64 = base64.b64encode(jpeg_data).decode('ascii')
        
        message = {
            "type": "frame",
            "data": frame_b64,
            "timestamp": time.time(),
        }
        if metadata:
            message["metadata"] = metadata
        
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self._clients:
            try:
                await client.send(message_str)
                self._frames_sent += 1
                self._bytes_sent += len(message_str)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        self._clients -= disconnected
    
    def _run_server(self):
        """Run the WebSocket server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        async def handler(websocket, path=None):
            """Handle a client connection."""
            if len(self._clients) >= self.config.max_clients:
                await websocket.close(1013, "Max clients reached")
                return
            
            self._clients.add(websocket)
            client_id = id(websocket)
            print(f"Client {client_id} connected ({len(self._clients)} total)")
            
            try:
                # Send initial info
                await websocket.send(json.dumps({
                    "type": "info",
                    "server": "quad-locomotion-stream",
                    "max_fps": self.config.max_fps,
                }))
                
                # Keep connection alive and handle messages
                async for message in websocket:
                    # Handle client commands if needed
                    try:
                        data = json.loads(message)
                        if data.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                    except:
                        pass
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self._clients.discard(websocket)
                print(f"Client {client_id} disconnected ({len(self._clients)} total)")
        
        async def serve():
            async with websocket_serve(
                handler,
                self.config.host,
                self.config.port,
            ):
                while self._running:
                    await asyncio.sleep(0.1)
        
        try:
            self._loop.run_until_complete(serve())
        except:
            pass
        finally:
            self._loop.close()
    
    @property
    def num_clients(self) -> int:
        """Number of connected clients."""
        return len(self._clients)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "clients": len(self._clients),
            "frames_sent": self._frames_sent,
            "bytes_sent": self._bytes_sent,
            "uptime": elapsed,
            "avg_fps": self._frames_sent / elapsed if elapsed > 0 else 0,
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
        return False
