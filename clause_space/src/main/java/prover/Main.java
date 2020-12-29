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
    System.out.println("file                                     clauses sat   processed   time");
    var solved = 0;
    for (var file : files) {
      System.out.printf("%-40s", file);
      status = null;
      var clauses = TptpParser.read(file);
      System.out.printf(" %7d", clauses.size());
      var start = System.currentTimeMillis();
      Superposition.timeout = start + 60_000;
      var result = Superposition.satisfiable(clauses);
      if (result == null) {
        System.out.print("      ");
      } else {
        System.out.print(result ? " sat  " : " unsat");
        if (result != status) {
          throw new IllegalStateException();
        }
        solved++;
      }
      System.out.printf(
          " %9d %6d\n", Superposition.processed.size(), System.currentTimeMillis() - start);
    }
    System.out.printf(
        "solved %d/%d (%f%%)\n", solved, files.length, solved * 100 / (double) files.length);
  }
}
