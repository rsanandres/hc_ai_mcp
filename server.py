#!/usr/bin/env python3
"""HC-AI MCP Server - Main entry point.

A Model Context Protocol (MCP) server that exposes healthcare AI tools
for querying patient data, reranking documents, and managing embeddings.

Usage:
    # Run with stdio transport (for Claude Desktop, Cursor)
    python server.py
    
    # Run with HTTP transport
    python server.py --transport http --port 8000
    
    # Use a custom config file
    python server.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure package is importable
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from dotenv import load_dotenv

# Load .env from this directory
load_dotenv(package_dir / ".env")


def log(message: str) -> None:
    """Log message to stderr (stdout is reserved for MCP protocol)."""
    print(message, file=sys.stderr)


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="HC-AI MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                          # Run with stdio transport
  python server.py --transport http         # Run with HTTP transport
  python server.py --config custom.yaml     # Use custom config file
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport type (overrides config.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0)",
    )
    
    args = parser.parse_args()
    
    # Import after argument parsing for faster --help
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        log(f"Error: MCP package not installed. Run: pip install mcp")
        log(f"Details: {e}")
        sys.exit(1)
    
    from config import load_config, get_server_config
    from tools import register_agent_tools, register_retrieval_tools, register_embeddings_tools
    
    # Load configuration
    config = load_config(args.config)
    server_config = get_server_config(config)
    
    # Determine transport
    transport = args.transport or server_config.get("transport", "stdio")
    
    # Create MCP server
    server_name = server_config.get("name", "HC-AI MCP Server")
    mcp = FastMCP(name=server_name)
    
    # Register tools based on configuration
    log(f"Starting {server_name}...")
    log(f"Transport: {transport}")
    log(f"Registering tools...")
    
    register_agent_tools(mcp, config)
    register_retrieval_tools(mcp, config)
    register_embeddings_tools(mcp, config)
    
    # Count enabled tools
    enabled_count = sum(
        1 for tool_name, tool_config in config.get("tools", {}).items()
        if tool_config.get("enabled", False)
    )
    log(f"Registered {enabled_count} tools")
    
    # Run server
    if transport == "streamable-http":
        # Set host/port via environment variables for uvicorn
        os.environ["UVICORN_HOST"] = args.host
        os.environ["UVICORN_PORT"] = str(args.port)
        log(f"Running on http://{args.host}:{args.port}")
        # For streamable-http, we need to run the ASGI app directly
        try:
            import uvicorn
            app = mcp.streamable_http_app()
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            log("Error: uvicorn is required for HTTP transport. Run: pip install uvicorn")
            sys.exit(1)
    else:
        log("Running with stdio transport")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
