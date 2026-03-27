@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d %~dp0
cl /O2 /EHsc slab_reader_bench.cpp /Fe:slab_reader_bench.exe
echo BUILD DONE
