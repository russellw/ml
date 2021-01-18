package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public final class Main {
  private enum Language {
    DIMACS,
    TPTP,
  }

  private static final class Summary {
    final String name;
    final double rating;
    final SZS expected;
    final SZS result;
    final long time;

    private Summary(Problem problem) {
      this.name = Etc.baseName(problem.file);
      this.rating = problem.rating;
      this.expected = problem.expected;
      this.result = problem.result;
      this.time = System.currentTimeMillis() - problem.startTime;
    }
  }

  public static List<Clause> memo;
  private static List<String> files = new ArrayList<>();
  private static Language language;
  private static long timeout = 300_000;
  private static int clauseLimit = 1000000;
  private static final String STDIN = "stdin";

  private static String optArg(String[] args, int i) throws IOException {
    if (i + 1 >= args.length) throw new IOException(args[i] + ": argument expected");
    return args[i + 1];
  }

  private static void args(String[] args) throws IOException {
    for (var i = 0; i < args.length; i++) {
      var arg = args[i];
      if (arg.isBlank()) continue;
      switch (arg.charAt(0)) {
        case '#':
          continue;
        case '-':
          break;
        default:
          if ("lst".equals(Etc.extension(arg))) {
            args(Files.readAllLines(Path.of(arg), StandardCharsets.UTF_8).toArray(new String[0]));
            continue;
          }
          files.add(arg);
          continue;
      }
      if ("-".equals(arg)) {
        files.add(STDIN);
        continue;
      }
      var opt = arg;
      while (opt.charAt(0) == '-') opt = opt.substring(1);
      String optArg = null;
      var j = 0;
      while (j < opt.length() && (Character.isLetter(opt.charAt(j)) || (opt.charAt(j) == '-'))) j++;
      if (j < opt.length()) {
        switch (opt.charAt(j)) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
            optArg = opt.substring(j);
            opt = opt.substring(0, j);
            break;
          case ':':
          case '=':
            optArg = opt.substring(j + 1);
            opt = opt.substring(0, j);
            break;
        }
      }
      switch (opt) {
        case "?":
        case "h":
        case "help":
          help();
          System.exit(0);
        case "T":
        case "cpu-limit":
          {
            if (optArg == null) optArg = optArg(args, i++);
            var seconds = Double.parseDouble(optArg);
            new Timer()
                .schedule(
                    new TimerTask() {
                      @Override
                      public void run() {
                        System.exit(7);
                      }
                    },
                    (long) (seconds * 1000));
            break;
          }
        case "V":
        case "show-version":
        case "showversion":
        case "version":
          System.out.printf(
              "Prover %s, %s\n",
              Objects.toString(version(), "[unknown version, not running from jar]"),
              System.getProperty("java.class.path"));
          System.out.printf(
              "%s, %s, %s\n",
              System.getProperty("java.vm.name"),
              System.getProperty("java.vm.version"),
              System.getProperty("java.home"));
          System.out.printf(
              "%s, %s, %s\n",
              System.getProperty("os.name"),
              System.getProperty("os.version"),
              System.getProperty("os.arch"));
          System.exit(0);
        case "dimacs":
          language = Language.DIMACS;
          break;
        case "delete-bad-limit":
        case "c":
          if (optArg == null) optArg = optArg(args, i++);
          clauseLimit = Integer.parseInt(optArg);
          break;
        case "t":
        case "soft-cpu-limit":
          {
            if (optArg == null) optArg = optArg(args, i++);
            var seconds = Double.parseDouble(optArg);
            timeout = (long) (seconds * 1000);
            break;
          }
        case "tptp":
          language = Language.TPTP;
          break;
        default:
          throw new IOException(arg + ": unknown option");
      }
    }
  }

  private Main() {}

  private static Language language(String file) throws IOException {
    if (language != null) return language;
    switch (Etc.extension(file)) {
      case "ax":
      case "p":
        return Language.TPTP;
      case "cnf":
        return Language.DIMACS;
    }
    throw new IOException(file + ": language not specified");
  }

  public static void main(String[] args) throws IOException {
    args(args);
    if (files.isEmpty()) {
      help();
      return;
    }
    var startTime = System.currentTimeMillis();
    var summaries = new ArrayList<Summary>();

    for (var file : files) {
      memo = null;
      var name = Etc.baseName(file);
      System.out.print(name);

      // Read
      var problem = new Problem(file);
      try {
        var stream = System.in;
        if (!file.equals(STDIN)) stream = new FileInputStream(file);
        switch (language(file)) {
          case TPTP:
            TptpParser.read(problem, stream);
            break;
          case DIMACS:
            DimacsParser.read(problem, stream);
            break;
          default:
            throw new IllegalStateException();
        }
      } catch (InappropriateException e) {
        System.out.println(",Inappropriate");
        continue;
      }

      // Solve
      problem.solve(clauseLimit, timeout);
      if (problem.result == SZS.Timeout) {
        System.out.println();
        continue;
      }

      // Result
      System.out.printf(
          "%s,%d,%.3f\n",
          problem.result,
          problem.iterations,
          (System.currentTimeMillis() - problem.startTime) * 0.001);
    }
    System.exit(0);

    for (var file : files) {
      System.out.println(file);
      memo = null;
      var name = Etc.baseName(file);

      // Read
      var problem = new Problem(file);
      try {
        var stream = System.in;
        if (!file.equals(STDIN)) stream = new FileInputStream(file);
        switch (language(file)) {
          case TPTP:
            TptpParser.read(problem, stream);
            break;
          case DIMACS:
            DimacsParser.read(problem, stream);
            break;
          default:
            throw new IllegalStateException();
        }
      } catch (InappropriateException e) {
        System.out.println("% SZS status Inappropriate for " + name);
        System.out.println();
        continue;
      }

      // Solve
      problem.solve(clauseLimit, timeout);

      // Result
      System.out.printf("%% SZS status %s for %s\n", problem.result, name);
      if (problem.refutation != null) {
        TptpPrinter.proof(problem.refutation);
      }
      System.out.printf(
          "%% %.3f seconds\n", (System.currentTimeMillis() - problem.startTime) * 0.001);
      problem.startTime = System.currentTimeMillis();

      // Solve
      memo = new ArrayList<>();
      for (var f : problem.refutation.proof())
        if (f instanceof Clause) {
          var c = (Clause) f;
          c = c.memo();
          memo.add(c);
        }
      problem.solve2(clauseLimit, timeout);

      // Result
      System.out.printf("%% SZS status %s for %s\n", problem.result, name);
      if (problem.refutation != null) {
        TptpPrinter.proof(problem.refutation);
      }

      // Statistics
      summaries.add(new Summary(problem));
      System.out.printf(
          "%% %.3f seconds\n", (System.currentTimeMillis() - problem.startTime) * 0.001);
      System.out.println();
    }
    if (summaries.isEmpty()) return;

    // Report
    summaries.sort(Comparator.comparingDouble((Summary o) -> o.rating).thenComparing(o -> o.name));
    try (var writer = new PrintWriter("/t/a.csv")) {
      for (var summary : summaries)
        writer.printf(
            "%s,%s,%s,%f,%d\n",
            summary.name, summary.expected, summary.result, summary.rating, summary.time);
    }

    // Overall
    var solved = Etc.count(summaries, summary -> Problem.solved(summary.result));
    System.out.printf(
        "Solved %d/%d (%f%%)\n",
        solved, summaries.size(), solved * 100 / (double) summaries.size());
    System.out.printf("%.3f seconds\n", (System.currentTimeMillis() - startTime) * 0.001);
  }

  private static void help() {
    System.out.println("Usage: prover [options] file");
    System.out.println();
    System.out.println("General options:");
    System.out.println("-help       Show help");
    System.out.println("-version    Show version");
    System.out.println();
    System.out.println("Input:");
    System.out.println("-dimacs     DIMACS format");
    System.out.println("-tptp       TPTP format");
    System.out.println("-           Read from stdin");
    System.out.println();
    System.out.println("Resources:");
    System.out.println("-c n        Clause limit, default 1000000");
    System.out.println("            Passive clauses over this limit will be discarded");
    System.out.println("-T seconds  Hard timeout");
    System.out.println("-t seconds  Soft timeout, default 300, 0=none");
    System.out.println("            Seconds can be floating point");
  }

  private static String version() throws IOException {
    var properties = new Properties();
    var stream =
        Main.class
            .getClassLoader()
            .getResourceAsStream("META-INF/maven/prover/prover/pom.properties");
    if (stream == null) return null;
    properties.load(stream);
    return properties.getProperty("version");
  }
}
