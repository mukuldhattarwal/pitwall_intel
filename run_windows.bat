@echo off
echo.
echo  ================================================
echo   PITWALL INTEL — Starting Up
echo  ================================================
echo.
echo  Installing dependencies...
pip install -r requirements.txt
echo.
echo  Launching app...
streamlit run app.py
pause
