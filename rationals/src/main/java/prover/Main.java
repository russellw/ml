package prover;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.function.Predicate;

public final class Main {
  public static boolean verbose;
  public static int clauseLimit;
  public static Language language;
  private static List<Path> paths = new ArrayList<>();
  private static String[] secondProver;
  private static String listFile;
  private static boolean minify;
  private static boolean optimize;
  private static boolean verify;
  public static long timeout;

  private Main() {}

  private static void args(String[] args) throws IOException {
    for (var i = 0; i < args.length; i++) {
      var arg = args[i];
      if (arg.isBlank()) {
        continue;
      }
      switch (arg.charAt(0)) {
        case '#':
          continue;
        case '-':
          break;
        default:
          if ("lst".equals(Util.extension(arg))) {
            if (listFile == null) {
              listFile = Util.removeExtension(arg);
            }
            args(Files.readAllLines(Path.of(arg), StandardCharsets.UTF_8).toArray(new String[0]));
            continue;
          }
          paths.add(Path.of(arg));
          continue;
      }
      if ("-".equals(arg)) {
        paths.add(Util.stdin);
        continue;
      }
      var opt = arg;
      while (opt.charAt(0) == '-') {
        opt = opt.substring(1);
      }
      String optArg = null;
      var j = 0;
      while ((j < opt.length()) && (Character.isLetter(opt.charAt(j)) || (opt.charAt(j) == '-'))) {
        j++;
      }
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
          throw new IllegalStateException();
        case "T":
          {
            if (optArg == null) {
              optArg = optArg(args, i++);
            }
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
          {
            var version = version();
            System.out.printf(
                "Prover %s, %s\n",
                Objects.toString(version, "[unknown version, not running from jar]"),
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
            throw new IllegalStateException();
          }
        case "clause-limit":
          {
            if (optArg == null) {
              optArg = optArg(args, i++);
            }
            clauseLimit = Integer.parseInt(optArg);
            break;
          }
        case "dimacs":
          language = Language.DIMACS;
          break;
        case "minify":
          if (optArg == null) {
            optArg = optArg(args, i++);
          }
          secondProver = optArg.split("\\s+");
          minify = true;
          break;
        case "ops-cons":
          if (optArg == null) {
            optArg = optArg(args, i++);
          }
          Subsumption.limitOpsConsidered = Integer.parseInt(optArg);
          break;
        case "ops-used":
          if (optArg == null) {
            optArg = optArg(args, i++);
          }
          Subsumption.limitOpsUsed = Integer.parseInt(optArg);
          break;
        case "optimize":
          optimize = true;
          break;
        case "subs-steps":
          if (optArg == null) {
            optArg = optArg(args, i++);
          }
          Subsumption.limitSteps = Integer.parseInt(optArg);
          break;
        case "t":
          {
            if (optArg == null) {
              optArg = optArg(args, i++);
            }
            var seconds = Double.parseDouble(optArg);
            timeout = (long) (seconds * 1000);
            break;
          }
        case "tptp":
          language = Language.TPTP;
          break;
        case "v":
        case "verbose":
          try {
            Files.createDirectory(Path.of("reports"));
          } catch (FileAlreadyExistsException ignored) {
          }
          verbose = true;
          break;
        case "verify":
          if (optArg == null) {
            optArg = optArg(args, i++);
          }
          secondProver = optArg.split("\\s+");
          verify = true;
          break;
        default:
          throw new IOException(arg + ": unknown option");
      }
    }
  }

  private static boolean fails(List<Clause> clauses) {
    if (clauses.isEmpty()) {
      return false;
    }
    for (var c : clauses) {
      if (c.isFalse()) {
        return false;
      }
    }
    System.out.println();
    System.out.printf("%d clauses, trying to manifest failure\n", clauses.size());
    var first = Problem.solve(clauses);
    try {
      var second = TptpPrinter.secondOpinion(secondProver, clauses);
      Util.printFile("/tmp/in.p");
      String vs;
      if (first.equals(second)) {
        vs = "==";
      } else if (first.compatible(second)) {
        vs = "~=";
      } else {
        vs = "!=";
      }
      System.out.printf("%s %s %s\n", first, vs, second);
      return !first.compatible(second);
    } catch (IOException | InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private static String format(Duration duration) {
    return String.format(
        "%3d:%02d:%02d",
        duration.toHoursPart(), duration.toMinutesPart(), duration.toSecondsPart());
  }

  private static void help() {
    System.out.println("Usage: prover [options] file");
    System.out.println();
    System.out.println("General options:");
    System.out.println("-help                    Show help");
    System.out.println("-version                 Show version");
    System.out.println();
    System.out.println("Input:");
    System.out.println("-dimacs                  DIMACS format");
    System.out.println("-tptp                    TPTP format");
    System.out.println("-                        Read from stdin");
    System.out.println();
    System.out.println("Subsumption parameters:");
    System.out.println("-ops-cons n              Operators considered for filtering");
    System.out.println("-ops-used n              Operators used for filtering");
    System.out.println("-subs-steps n            Maximum number of steps");
    System.out.println();
    System.out.println("Resources:");
    System.out.println("-clause-limit n          Limit number of clauses in memory");
    System.out.println("-T seconds               Hard timeout");
    System.out.println("-t seconds               Soft timeout");
    System.out.println("                         Seconds can be floating point");
    System.out.println();
    System.out.println("Output:");
    System.out.println("-optimize                Seek optimal parameter values");
    System.out.println("-verify \"second prover\"  Try to verify results");
    System.out.println("                         with a second prover checking each step");
    System.out.println("-minify \"second prover\"  Look for minimal version");
    System.out.println("                         of problem that manifests a bug");
    System.out.println("-verbose                 Extra output");
    System.out.println("                         emits reports/*.html");
    System.exit(0);
  }

  public static void main(String[] args) throws InterruptedException {
    if (args.length == 0) {
      help();
    }
    try {
      args(args);
      if (paths.isEmpty()) {
        paths.add(Util.stdin);
      }
      if (optimize) {
        optimize();
        return;
      }
      for (var path : paths) {
        var file = path.getFileName().toString();

        // Parse
        Problem problem;
        try {
          problem = Problem.read(path);
          if (verbose) {
            for (var i = 0; (i < problem.header.size()) && (i < 66); i++) {
              System.out.println(problem.header.get(i));
            }
          }
        } catch (InappropriateException e) {
          System.out.println("% SZS status Inappropriate for " + file);
          if (paths.size() > 1) {
            System.out.println();
          }
          continue;
        }

        // Solve
        problem.solve();

        // Report
        switch (Problem.language(path)) {
          case DIMACS:
            DimacsPrinter.print(problem.result);
            break;
          case TPTP:
            System.out.printf("%% SZS status %s for %s\n", problem.result, file);
            if (problem.superposition.conclusion != null) {
              System.out.println("% SZS output start CNFRefutation for " + file);
              TptpPrinter.out.print(problem.superposition.conclusion.proof());
              System.out.println("% SZS output end CNFRefutation for " + file);
              if (verify) {
                TptpPrinter.verify(secondProver, problem.superposition.conclusion);
              }
            }
            break;
        }
        if (verbose) {
          problem.print(file);
        }

        // Check
        if (!problem.result.compatible(problem.expected)) {
          if (minify) {
            System.out.println("Problem header said " + problem.expected);
            System.out.println("Minifying");
            var clauses = minify(problem.clauses, Main::fails);
            System.out.println();
            System.out.println("Minimal test case:");
            TptpPrinter.out.print(clauses);
            System.out.println();
            return;
          }
          throw new IllegalStateException("Problem header said " + problem.expected);
        }
        if (paths.size() > 1) {
          System.out.println();
        }
      }

      // Blank line between multiple problems
      if (verbose && (paths.size() > 1)) {
        Problem.printSummary(listFile);
      }
    } catch (IOException e) {
      System.err.println(e.getMessage());
      System.exit(1);
    }
  }

  private static List<Clause> minify(Clause c) {
    var r = new ArrayList<Clause>();
    for (var i = 0; i < c.literals.length; i++) {
      r.add(
          new Clause(
              Term.remove(c.literals, i),
              (i < c.negativeSize) ? c.negativeSize - 1 : c.negativeSize));
    }
    return r;
  }

  private static List<List<Clause>> minify(List<Clause> clauses) {
    var rs = new ArrayList<List<Clause>>();
    for (var i = 0; i < clauses.size(); i++) {
      rs.add(Util.remove(clauses, i));
      for (var c : minify(clauses.get(i))) {
        rs.add(Util.replace(clauses, i, c));
      }
    }
    return rs;
  }

  private static List<Clause> minify(List<Clause> clauses, Predicate<List<Clause>> f) {
    if (!f.test(clauses)) {
      throw new IllegalArgumentException();
    }
    shrink:
    for (; ; ) {
      for (var clauses1 : minify(clauses)) {
        if (f.test(clauses1)) {
          clauses = clauses1;
          continue shrink;
        }
      }
      return clauses;
    }
  }

  private static String optArg(String[] args, int i) throws IOException {
    if (i + 1 >= args.length) {
      throw new IOException(args[i] + ": argument expected");
    }
    return args[i + 1];
  }

  private static void optimize() {
    var params = new Optimizer.Param[] {};
    System.out.print("   ");
    for (var param : params) {
      System.out.printf(" %7s", param.name());
    }
    System.out.println("  solved      time     total");
    Optimizer.optimize(
        params,
        () -> {
          try {
            var begin = System.currentTimeMillis();
            var solved = 0;
            var time = 0L;
            for (var path : paths) {

              // Parse
              Problem problem;
              try {
                problem = Problem.read(path);
              } catch (InappropriateException e) {
                continue;
              }

              // Solve
              problem.solve();

              // Check
              if (!problem.result.compatible(problem.expected)) {
                throw new IllegalStateException("Problem header said " + problem.expected);
              }

              // Score
              if (problem.result.solved()) {
                solved++;
                time += problem.timeEnd - problem.timeBegin;
              }
            }
            System.out.printf(
                " %7d %s %s\n",
                solved,
                format(Duration.ofMillis(time)),
                format(Duration.ofMillis(System.currentTimeMillis() - begin)));
            return solved - time / 1e9;
          } catch (IOException e) {
            System.err.println(e.getMessage());
            System.exit(1);
            throw new IllegalStateException();
          }
        });
  }

  public static String version() throws IOException {
    var properties = new Properties();
    var stream =
        Main.class
            .getClassLoader()
            .getResourceAsStream("META-INF/maven/prover/prover/pom.properties");
    if (stream == null) {
      return null;
    }
    properties.load(stream);
    return properties.getProperty("version");
  }
}
