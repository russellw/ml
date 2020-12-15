call mvn test
if errorlevel 1 goto :eof
java -cp target/classes -ea specs/Main %*
