python -m http.server 8080
# With OSMesa for headless rendering
MUJOCO_GL=osmesa python examples/test_streaming.py

# Then open viewer.html in a browser (served at http://localhost:8080/viewer.html)