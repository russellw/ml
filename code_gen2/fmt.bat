clang-format -i -style=file *.cc src\*.h src\*.cc test\*.cc
if %errorlevel% neq 0 goto :eof

black .
if %errorlevel% neq 0 goto :eof

git diff
