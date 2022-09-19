black .
if %errorlevel% neq 0 goto :eof

python fmt.py
if %errorlevel% neq 0 goto :eof

git diff
