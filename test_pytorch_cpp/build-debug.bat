if not exist torch_cpu.dll copy C:\libtorch-debug\lib\torch_cpu.dll
if errorlevel 1 goto :eof

if not exist c10.dll copy C:\libtorch-debug\lib\c10.dll
if errorlevel 1 goto :eof

if not exist fbgemm.dll copy C:\libtorch-debug\lib\fbgemm.dll
if errorlevel 1 goto :eof

if not exist asmjit.dll copy C:\libtorch-debug\lib\asmjit.dll
if errorlevel 1 goto :eof

cl /IC:\libtorch-debug\include /IC:\libtorch-debug\include\torch\csrc\api\include autograd.cpp C:\libtorch-debug\lib\*.lib
if errorlevel 1 goto :eof

autograd
