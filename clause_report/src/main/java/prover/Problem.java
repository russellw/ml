package prover;

import java.util.ArrayList;

public final class Problem {
  public SZS expected;
  public ArrayList<Clause> clauses = new ArrayList<>();
  public Clause proof;
  public SZS result;
}
