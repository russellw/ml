cl /DDEBUG /EHsc /IC:\ml\code_gen2 /WX /Zi C:\ml\code_gen2\test.cc C:\ml\code_gen2\lo\*.cc dbghelp.lib
if %errorlevel% neq 0 goto :eof

test
