clang-format -i -style=file lib\*.h lib\*.cc compiler\*.cc
if %errorlevel% neq 0 goto :eof

black .
if %errorlevel% neq 0 goto :eof

git diff
