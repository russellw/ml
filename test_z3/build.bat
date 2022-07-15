rem examples.cpp tested with
rem z3-4.9.1-x64-win.zip

if not exist libz3.dll copy C:\z3\bin\libz3.dll
if errorlevel 1 goto :eof

cl /EHsc /IC:\z3\include examples.cpp C:\z3\bin\libz3.lib
if errorlevel 1 goto :eof

examples
