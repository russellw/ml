package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

public final class Main {
  public static Boolean status;

  private Main() {}

  public static void main(String[] args) throws IOException {
    if (args[0].endsWith(".csv")) {
      RegressionTree.test(args[0]);
      return;
    }

    // files
    if (Files.isDirectory(Path.of(args[0]))) {
      args =
          Files.walk(Path.of(args[0]))
              .filter(path -> !Files.isDirectory(path))
              .map(Path::toString)
              .toArray(String[]::new);
    } else if (args[0].endsWith(".lst")) {
      args = Files.readAllLines(Path.of(args[0]), StandardCharsets.UTF_8).toArray(new String[0]);
    }

    // vocabulary
    var counts = new HashMap<String, Integer>();
    for (var file : args) {
      var clauses = TptpParser.read(file);
      for (var c : clauses) {
        for (var a : c.literals) {
          a.walk(
              term -> {
                if (term instanceof Function) {
                  counts.merge(term.toString(), 1, Integer::sum);
                }
              });
        }
      }
    }
    var sorted = new ArrayList<>(counts.keySet());
    sorted.sort((s1, s2) -> counts.get(s2) - counts.get(s1));
    for (int i = 0; (i < 50) && (i < sorted.size()); i++) {
      var name = sorted.get(i);
      Function.COMMON_NAMES.put(name, new Variable(Type.NUMBER, name));
      System.out.printf("%-40s %7d\n", name, counts.get(name));
    }
    System.out.println(sorted.size() + " words");
    System.out.println();

    // main loop
    for (int i = 0; i < 1; i++) {

      // do
      System.out.println("file                                     clauses sat   processed   time");
      var records = new ArrayList<Record>();
      var solved = 0;
      for (var file : args) {

        // solve
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

        // records
        var distances = new HashMap<Clause, Integer>();
        if (Superposition.proof != null) {
          Superposition.proof.walk(distances::put);
          for (var c : Superposition.processed) {
            records.add(new Record(c.map(), new Number(distances.getOrDefault(c, 1_000_000))));
          }
        }
      }
      System.out.printf(
          "solved %d/%d (%f%%)\n", solved, args.length, solved * 100 / (double) args.length);
      System.out.println();

      // learn
      System.out.printf("training with %d records\n", records.size());
      var start = System.currentTimeMillis();
      Clause.cost = RegressionTree.of(records);
      System.out.printf("%f seconds\n", (System.currentTimeMillis() - start) / 1000.0);
      System.out.println();
    }
  }
}
