clang-format -i -style=file *.h *.cc lo\*.h lo\*.cc
if %errorlevel% neq 0 goto :eof

black .
if %errorlevel% neq 0 goto :eof

git diff
