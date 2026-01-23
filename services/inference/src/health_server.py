"""
Simple HTTP server for health checks in inference service
Runs in a separate thread to provide health endpoint
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import logging

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Handler for health check requests"""
    
    # Track service readiness
    service_ready = False
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            status = "healthy" if HealthCheckHandler.service_ready else "starting"
            status_code = 200 if HealthCheckHandler.service_ready else 503
            
            response = {
                "status": status,
                "service": "inference"
            }
            
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging to avoid clutter"""
        pass


def start_health_server(port=8080):
    """Start health check HTTP server in a separate thread"""
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    
    def serve():
        logger.info(f"Health check server started on port {port}")
        server.serve_forever()
    
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server


def set_service_ready():
    """Mark service as ready for health checks"""
    HealthCheckHandler.service_ready = True
    logger.info("Service marked as ready for health checks")
