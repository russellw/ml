package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class Main {
  public static Boolean status;

  public static void main(String[] args) throws IOException {
    assert args[0].endsWith(".lst");
    var files = Files.readAllLines(Path.of(args[0]), StandardCharsets.UTF_8).toArray(new String[0]);
    for (var file : files) {
      var clauses = TptpParser.read(file);
      System.out.println(clauses.size());
    }
  }
}
