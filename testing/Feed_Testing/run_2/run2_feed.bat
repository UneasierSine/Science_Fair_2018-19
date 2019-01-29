@echo off

cd /D %cd%\1
python Feed_Testing_2.py
cd /D ..

cd /D %cd%\2
python Feed_Testing_2.py
cd /D ..

cd /D %cd%\3
python Feed_Testing_2.py
cd /D ..

cd /D %cd%\4
python Feed_Testing_2.py
cd /D ..

cd /D %cd%\5
python Feed_Testing_2.py
cd /D ..