if "%VCINSTALLDIR%"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /DBENCHMARK /EHsc /Feayane /I\mpir /J /O2 *.cc \mpir\release.lib
if errorlevel 1 goto :eof
ayane
