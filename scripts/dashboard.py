#!/usr/bin/env python3
"""
Dashboard launcher for SIIC Vietnamese Emotion Detection System.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --port 8501
    python scripts/dashboard.py --dev
"""

import argparse
import subprocess
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser(description='Launch SIIC Dashboard')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port to run dashboard on (default: 8501)')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with debug logging')
    
    args = parser.parse_args()
    
    # Dashboard app path
    dashboard_path = os.path.join(project_root, 'dashboard', 'app.py')
    
    # Build streamlit command
    cmd = ['streamlit', 'run', dashboard_path, '--server.port', str(args.port)]
    
    if args.dev:
        cmd.extend(['--logger.level', 'debug'])
    else:
        cmd.extend(['--server.headless', 'true'])
    
    print(f"Starting SIIC Dashboard on http://localhost:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")

if __name__ == "__main__":
    main() 