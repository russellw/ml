pushd \mpir\msvc\vs19
call msbuild.bat gc DLL x64 Debug
call msbuild.bat gc LIB x64 Release
popd
copy \mpir\dll\x64\Debug\mpir.dll
