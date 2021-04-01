if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
move *.asm \temp
cl /Fa /Fea /Ilib /I\mpir /J /O2 *.c lib\*.c \mpir\release.lib
wc *.asm
