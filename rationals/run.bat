call mvn test
if errorlevel 1 goto :eof

rem https://www.baeldung.com/java-verbose-gc
rem https://stackoverflow.com/questions/7093710/java-need-verbose-gc-logging-to-separate-files
rem java -cp target/classes -ea -verbose:gc prover/Main -t 60 -v %*

del gc.log
java -Xlog:gc:gc.log -cp target/classes -ea prover/Main -clause-limit 1000000 -t 60 -v %*

rem java -Xmx1000m -ea -jar target\prover-1.0-SNAPSHOT.jar -v %*
