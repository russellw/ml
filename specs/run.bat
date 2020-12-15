call mvn package
if errorlevel 1 goto :eof
java -XX:MaxJavaStackTraceDepth=50 -Xss1m -ea -jar target/specs-1.0-SNAPSHOT-jar-with-dependencies.jar %*
