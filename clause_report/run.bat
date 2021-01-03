call mvn test
if errorlevel 1 goto :eof
java -XX:MaxJavaStackTraceDepth=1000000 -cp target/classes -ea prover/Main %*
