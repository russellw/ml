if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
move *.asm \temp
cl /DDEBUG /Fea /Ilib /I\mpir /J /MP4 /MTd /WX /Yustdafx.h /Zi *.c stdafx.obj \mpir\debug.lib dbghelp.lib >\temp\1
if errorlevel 1 goto :err
a %*
echo %errorlevel%
goto :eof

:err
head -n 50 \temp\1
