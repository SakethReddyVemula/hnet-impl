#!/bin/bash

# Activate the virtual environment
export PYTHONUTF8=1
source ~/santam-tok/hnet-venv/bin/activate

# Check if streamlit is installed, if not install it
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing..."
    pip3 install streamlit
fi

# Run the app
echo "Starting H-Net Visualizer..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
