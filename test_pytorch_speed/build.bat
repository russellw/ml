if not exist torch_cpu.dll copy C:\libtorch\lib\torch_cpu.dll
if not exist c10.dll copy C:\libtorch\lib\c10.dll
if not exist fbgemm.dll copy C:\libtorch\lib\fbgemm.dll
if not exist asmjit.dll copy C:\libtorch\lib\asmjit.dll
cl /IC:\libtorch\include /IC:\libtorch\include\torch\csrc\api\include /O2 vecs.cpp C:\libtorch\lib\*.lib
if errorlevel 1 goto :eof
