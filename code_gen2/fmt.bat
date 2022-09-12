black .
if %errorlevel% neq 0 goto :eof

clang-format -i -style=file *.h *.cc
if %errorlevel% neq 0 goto :eof

git diff
