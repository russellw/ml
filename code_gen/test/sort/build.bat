cl /c %~dp0str.c
if errorlevel 1 goto :eof

link /dll str.obj
if errorlevel 1 goto :eof

cl /c %~dp0vec.c
if errorlevel 1 goto :eof

link /dll vec.obj
if errorlevel 1 goto :eof

cl /c %~dp0readfile.c
if errorlevel 1 goto :eof

link /dll readfile.obj str.lib vec.lib
if errorlevel 1 goto :eof

cl %~dp0sort.c str.lib readfile.lib
if errorlevel 1 goto :eof

sort %~dp0test.txt
