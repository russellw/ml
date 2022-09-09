cl /c *.c
if errorlevel 1 goto :eof

lib foo1.obj check1.obj
if errorlevel 1 goto :eof

lib /out:foo2.lib check2.obj foo2.obj
if errorlevel 1 goto :eof

lib /out:foo3.lib foo3.obj check3.obj
if errorlevel 1 goto :eof

link /verbose /verbose:lib main.obj foo1.lib foo2.lib foo3.lib|rg -w foo
if errorlevel 1 goto :eof

main
