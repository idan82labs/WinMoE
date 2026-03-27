Cache code is written. Compile and run it NOW.

Run this:
```
engine\cache\build_bench.bat
engine\cache\cache_bench.exe
```

If build_bench.bat fails, try directly:
```
powershell -Command "Start-Process cmd -ArgumentList '/c cd /d C:\Users\idant\flash-moe-windows-v2\flash-moe-windows-v2-claude-filesystem-autoresearch-loops\engine\cache && build_bench.bat > build.log 2>&1' -Wait"
type engine\cache\build.log
```

Report the hit rate numbers, then move to Component 4 (streaming scheduler).
