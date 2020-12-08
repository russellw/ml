package prover;

public enum SZS {
  ContradictoryAxioms,
  CounterEquivalent,
  CounterSatisfiable,
  EquiSatisfiable,
  Equivalent,
  Error,
  GaveUp,
  LogicalData,
  Open,
  ResourceOut,
  Satisfiable,
  Theorem,
  Timeout,
  Unknown,
  Unsatisfiable;

  public String abbreviation() {
    switch (this) {
      case ContradictoryAxioms:
        return "CAX";
      case CounterEquivalent:
        return "CEQ";
      case CounterSatisfiable:
        return "CSA";
      case EquiSatisfiable:
        return "ESA";
      case Equivalent:
        return "EQV";
      case Error:
        return "ERR";
      case GaveUp:
        return "GUP";
      case LogicalData:
        return "LDa";
      case Open:
        return "OPN";
      case ResourceOut:
        return "RSO";
      case Satisfiable:
        return "SAT";
      case Theorem:
        return "THM";
      case Timeout:
        return "TMO";
      case Unknown:
        return "UNK";
      case Unsatisfiable:
        return "UNS";
    }
    throw new IllegalStateException(toString());
  }

  public boolean compatible(SZS b) {
    if (b == null) {
      return true;
    }
    switch (this) {
      case ContradictoryAxioms:
      case Theorem:
      case Unsatisfiable:
        switch (b) {
          case ContradictoryAxioms:
          case Theorem:
          case Unsatisfiable:
            return true;
        }
        return false;
      case CounterSatisfiable:
      case Satisfiable:
        switch (b) {
          case CounterSatisfiable:
          case Satisfiable:
            return true;
        }
        return false;
    }
    return true;
  }

  public boolean solved() {
    switch (this) {
      case ContradictoryAxioms:
      case CounterSatisfiable:
      case Satisfiable:
      case Theorem:
      case Unsatisfiable:
        return true;
    }
    return false;
  }
}
