rem Run this in the root of the Python source tree
rem e.g. in C:\Python-3.10.4

git init
if errorlevel 1 goto :eof

copy %~dp0..\.gitignore
if errorlevel 1 goto :eof

git add .
if errorlevel 1 goto :eof

git commit -m "Initial commit"
if errorlevel 1 goto :eof

git apply %~dp0fix.patch
if errorlevel 1 goto :eof

git commit -a -m "Patch for clang"
if errorlevel 1 goto :eof

rem -r tells build.bat to do a full rebuild
rem so that if a previous build was done
rem we still get a log of the whole process
rem -fl tells MSBuild to write a log of what it does

rem for debug build, add this option:
rem -c Debug
call PCbuild\build.bat -p x64 -r -fl %*
if errorlevel 1 goto :eof
