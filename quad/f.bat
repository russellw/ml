black  .
if %errorlevel% neq 0 goto :eof

isort .
if %errorlevel% neq 0 goto :eof

python akfmt.py .
if %errorlevel% neq 0 goto :eof

git diff
