if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
move *.asm \temp
cl /DDEBUG /Fea /Ilib /I\mpir /J /MP4 /MTd /W2 /WX /Zi *.c lib\*.c \mpir\debug.lib dbghelp.lib
if errorlevel 1 goto :eof
a %*
echo %errorlevel%
