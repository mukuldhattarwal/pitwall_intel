#!/bin/bash
echo ""
echo " ================================================"
echo "  PITWALL INTEL — Starting Up"
echo " ================================================"
echo ""
echo " Installing dependencies..."
pip install -r requirements.txt
echo ""
echo " Launching app at http://localhost:8501"
streamlit run app.py
