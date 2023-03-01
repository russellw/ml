black  .
if %errorlevel% neq 0 goto :eof

isort .
if %errorlevel% neq 0 goto :eof

copy *.py %tmp%
call defsort .
if %errorlevel% neq 0 goto :eof

git diff
