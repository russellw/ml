pushd \mpir\msvc\vs19
call msbuild.bat core2 DLL x64 Debug +tests
call msbuild.bat core2 LIB x64 Release +tests
popd
copy \mpir\dll\x64\Debug\mpir.dll
