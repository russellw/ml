if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
move *.asm \temp
cl /DDEBUG /EHsc /Feayane /I\mpir /J /MP4 /MTd /WX /Yustdafx.h /Zi *.cc stdafx.obj \mpir\debug.lib dbghelp.lib >\temp\1
if errorlevel 1 goto :err
ayane %*
echo %errorlevel%
goto :eof

:err
head -n 50 \temp\1
