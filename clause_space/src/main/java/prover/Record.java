package prover;

import java.util.Map;

public final class Record {
  public final Map<Variable, Term> inputs;
  public final Term output;

  public Record(Map<Variable, Term> inputs, Term output) {
    this.inputs = inputs;
    this.output = output;
  }
}
