@echo off

cmd /C conda activate tf_gpu
SET root=C:\Users\panka\Desktop\Science_Fair_2018-19\testing

python "%root%\Normal_Testing\run_3\1\Normal_Testing_3.py"

python "%root%\Feed_Testing\run_4\1\Feed_Testing_4.py"
python "%root%\Normal_Testing\run_4\1\Normal_Testing_4.py"