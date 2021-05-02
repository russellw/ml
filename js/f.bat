call npx prettier --write .
if errorlevel 1 goto :eof

call node cleanjs.js .
if errorlevel 1 goto :eof

git diff
