black  .
if %errorlevel% neq 0 goto :eof

isort .
if %errorlevel% neq 0 goto :eof

call defsort .
if %errorlevel% neq 0 goto :eof

git diff
