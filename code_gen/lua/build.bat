if not exist "%1\src\Makefile" goto :eof

md temp-lua
copy %1\src temp-lua
if errorlevel 1 goto :eof
del temp-lua\luac.*

cl /Felua-cl0 temp-lua\*.c
if errorlevel 1 goto :eof

rem in benchmarks, /O1 was the fastest of the Microsoft-compiled interpreters
cl /Felua-cl1 /O1 temp-lua\*.c
if errorlevel 1 goto :eof

cl /Felua-cl2 /O2 temp-lua\*.c
if errorlevel 1 goto :eof

cl /Felua-cls /Os temp-lua\*.c
if errorlevel 1 goto :eof

clang -olua-clang0.exe temp-lua\*.c
if errorlevel 1 goto :eof

clang -O1 -olua-clang1.exe temp-lua\*.c
if errorlevel 1 goto :eof

rem in benchmarks, clang -O2 and -O3 were, to within measurement accuracy,
rem as fast as each other and the Microsoft-compiled interpreter
clang -O2 -olua-clang2.exe temp-lua\*.c
if errorlevel 1 goto :eof

clang -O3 -olua-clang3.exe temp-lua\*.c
if errorlevel 1 goto :eof

clang -Os -olua-clangs.exe temp-lua\*.c
if errorlevel 1 goto :eof

del *.bc
for %%x in (temp-lua\*.c) do clang -O3 -c -emit-llvm %%x
if errorlevel 1 goto :eof
olivine *.bc
if errorlevel 1 goto :eof
clang -O3 -olua-inline.exe a.ll
if errorlevel 1 goto :eof
