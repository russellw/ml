if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
move *.asm \temp
cl /EHsc /Fa /Feayane /I\mpir /J /O2 *.cc \mpir\release.lib
wc *.asm
