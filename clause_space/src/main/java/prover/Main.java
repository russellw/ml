package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class Main {
  public static Boolean status;

  public static void main(String[] args) throws IOException {
    var arg = args[0];
    String[] files;
    if (arg.endsWith(".lst"))
      files = Files.readAllLines(Path.of(arg), StandardCharsets.UTF_8).toArray(new String[0]);
    else files = new String[] {arg};
    for (var file : files) {
      System.out.print(file);
      status = null;
      var clauses = TptpParser.read(file);
      System.out.print("\t" + clauses.size());
      for (var i = 0; i < 2; i++) {
        clauses = Superposition.expand(clauses);
        System.out.print("\t" + clauses.size());
      }
      System.out.println();
    }
  }
}
