cl /c *.cc
if errorlevel 1 goto :eof

link /dll /out:sqrt-plain.dll sqrt-plain
if errorlevel 1 goto :eof

link /dll /out:sqrt-ctor.dll sqrt-ctor
if errorlevel 1 goto :eof

link /dll /out:sqrt-dllmain.dll sqrt-dllmain
if errorlevel 1 goto :eof

link /dll /out:sqrt-load.dll sqrt-load
if errorlevel 1 goto :eof

cl main.cpp sqrt-plain.lib sqrt-ctor.lib sqrt-dllmain.lib
if errorlevel 1 goto :eof

main
