for %%x in (*.py) do black %%x
dir /b /s *.java >\temp\files.txt
java -jar /bin/google-java-format-1.8-all-deps.jar -i @\temp\files.txt
git diff
