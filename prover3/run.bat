call mvn package
if errorlevel 1 goto :eof
java -XX:MaxJavaStackTraceDepth=50 -Xss10m -ea -jar target/prover-1.0-SNAPSHOT-jar-with-dependencies.jar %*
