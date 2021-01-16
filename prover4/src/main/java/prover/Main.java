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
    final SZS result;
    final long time;

    private Summary(Problem problem) {
      this.name = Etc.baseName(problem.file);
      this.rating = problem.rating;
      this.result = problem.result;
      this.time = System.currentTimeMillis() - problem.startTime;
    }
  }

  private static String listFile;
  private static List<String> files = new ArrayList<>();
  private static Language language;
  private static long timeout = 60_000;
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
            if (listFile == null) listFile = arg;
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
          {
            if (optArg == null) optArg = optArg(args, i++);
            var seconds = Double.parseDouble(optArg);
            new Timer()
                .schedule(
                    new TimerTask() {
                      @Override
                      public void run() {
                        System.exit(1);
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
        case "t":
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
      problem.solve(timeout);

      // Result
      file = Etc.withoutDir(file);
      System.out.printf("%% SZS status %s for %s\n", problem.result, name);
      if (problem.refutation != null) {
        System.out.println("% SZS output start CNFRefutation for " + name);
        TptpPrinter.proof(problem.refutation);
        System.out.println("% SZS output end CNFRefutation for " + name);
      }

      // Statistics
      summaries.add(new Summary(problem));
      System.out.printf(
          "%% %.3f seconds\n", (System.currentTimeMillis() - problem.startTime) * 0.001);
      System.out.println();
    }

    // Report
    var writer = new PrintWriter("/t/a.csv");
    for (var summary : summaries) writer.printf("%s\n", summary.name);
    writer.close();

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
    System.out.println("-T seconds  Hard timeout");
    System.out.println("-t seconds  Soft timeout");
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
