black %~dp0..
if errorlevel 1 goto :eof

python %~dp0fmt.py -i %~dp0..
if errorlevel 1 goto :eof

git diff
