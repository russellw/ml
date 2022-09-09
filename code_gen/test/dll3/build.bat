cl /c *.c
if errorlevel 1 goto :eof

link /dll foo1.obj check1.obj
if errorlevel 1 goto :eof

link /dll /out:foo2.dll /dll check2.obj foo2.obj
if errorlevel 1 goto :eof

link /dll /out:foo3.dll foo3.obj check3.obj
if errorlevel 1 goto :eof

link main.obj foo1.lib foo2.lib foo3.lib
if errorlevel 1 goto :eof

main
