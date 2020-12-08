package prover;

public final class DimacsPrinter {
  private DimacsPrinter() {}

  public static void print(SZS szs) {
    switch (szs) {
      case Satisfiable:
        System.out.println("sat");
        break;
      case Unsatisfiable:
        System.out.println("unsat");
        break;
    }
  }
}
