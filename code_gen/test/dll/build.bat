cl /c square.c cube.c
if errorlevel 1 goto :eof

link /dll /out:poly.dll square.obj cube.obj
if errorlevel 1 goto :eof

cl main.c poly.lib
if errorlevel 1 goto :eof

main
