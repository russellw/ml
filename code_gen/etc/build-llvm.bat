rem https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-14.0.5.zip
rem https://llvm.org/docs/GettingStarted.html

cd \llvm-project
if errorlevel 1 goto :eof

md build
cd build
if errorlevel 1 goto :eof

"\Program Files\CMake\bin\cmake.exe" -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=1 ../llvm
if errorlevel 1 goto :eof

cmake --build .
if errorlevel 1 goto :eof

Debug\bin\clang.exe --version
