package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;

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
      var n = 0;
      var images = new HashSet<String>();
      for (var g : clauses) {
        var generated = Superposition.expand(clauses, g);
        n += generated.size();
        for (var c : generated) images.add(c.image());
      }
      System.out.print("\t" + n);
      System.out.print("\t" + images.size());
      System.out.println();
    }
  }
}
