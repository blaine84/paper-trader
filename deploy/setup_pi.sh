#!/bin/bash
# Paper Trader — Raspberry Pi Setup Script
# Run as: bash setup_pi.sh
# Tested on Raspberry Pi OS (Bookworm/Bullseye), Pi 3B+ or better

set -e

echo "=== Paper Trader Pi Setup ==="

# 1. System deps
echo "[1/6] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y python3 python3-pip python3-venv git

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_VERSION" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required. Current: 3.$PYTHON_VERSION"
    echo "On older Pi OS: sudo apt install python3.11"
    exit 1
fi
echo "  Python version OK: 3.$PYTHON_VERSION"

# 2. Clone or update repo
INSTALL_DIR="/home/pi/paper-trader"
if [ -d "$INSTALL_DIR" ]; then
    echo "[2/6] Updating existing install..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "[2/6] Cloning repo..."
    # Replace with your actual repo URL or copy files manually
    echo "  No git remote configured — copy files to $INSTALL_DIR manually"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 3. Virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."

# 4. Environment file
if [ ! -f .env ]; then
    echo "[4/6] Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "  *** ACTION REQUIRED ***"
    echo "  Edit /home/pi/paper-trader/.env and add your API keys:"
    echo "    nano /home/pi/paper-trader/.env"
    echo ""
else
    echo "[4/6] .env already exists, skipping."
fi

# 5. Create logs directory
mkdir -p logs
echo "[5/6] Logs directory ready."

# 6. Install systemd service
echo "[6/6] Installing systemd service..."
sudo cp deploy/paper-trader.service /etc/systemd/system/paper-trader.service
sudo systemctl daemon-reload
sudo systemctl enable paper-trader.service
echo "  Service installed and enabled on boot."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit your API keys:  nano /home/pi/paper-trader/.env"
echo "  2. Test single cycle:   cd /home/pi/paper-trader && source venv/bin/activate && python orchestrator.py once"
echo "  3. Start the service:   sudo systemctl start paper-trader"
echo "  4. Check logs:          sudo journalctl -u paper-trader -f"
echo "                          tail -f /home/pi/paper-trader/logs/service.log"
echo ""
echo "Service commands:"
echo "  sudo systemctl start paper-trader"
echo "  sudo systemctl stop paper-trader"
echo "  sudo systemctl restart paper-trader"
echo "  sudo systemctl status paper-trader"
