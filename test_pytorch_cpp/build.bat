if not exist torch_cpu.dll copy C:\libtorch\lib\torch_cpu.dll
if errorlevel 1 goto :eof

if not exist c10.dll copy C:\libtorch\lib\c10.dll
if errorlevel 1 goto :eof

if not exist fbgemm.dll copy C:\libtorch\lib\fbgemm.dll
if errorlevel 1 goto :eof

if not exist asmjit.dll copy C:\libtorch\lib\asmjit.dll
if errorlevel 1 goto :eof

cl /IC:\libtorch\include /IC:\libtorch\include\torch\csrc\api\include autograd.cpp C:\libtorch\lib\*.lib
if errorlevel 1 goto :eof

autograd
