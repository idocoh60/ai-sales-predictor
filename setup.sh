#!/bin/bash

echo "Creating virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

echo "Running the application..."
python app.py
