package prover;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public final class Summary {
  public final String name;
  public final int formulas;
  public final int clauses;
  public final int active;
  public final int passive;
  public final SZS expected;
  public final SZS result;
  public final double rating;
  public final long startTime;
  public final long endTime;
  List<Record> records = new ArrayList<>();

  public Summary(Problem problem) {
    name = Etc.baseName(problem.file());
    formulas = problem.formulas.size();
    clauses = problem.clauses.size();
    active = problem.superposition.active.size();
    passive = problem.superposition.passive.size();
    expected = problem.expected;
    result = problem.result;
    rating = problem.rating;
    startTime = problem.startTime;
    endTime = problem.endTime;
    records = problem.records;
  }

  public static void write(String name, List<Summary> summaries) throws FileNotFoundException {}
}
