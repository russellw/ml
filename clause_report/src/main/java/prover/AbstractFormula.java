package prover;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.function.Consumer;

public abstract class AbstractFormula {
  public Object name;
  public String file;
  public final Inference inference;
  public final AbstractFormula[] from;

  protected AbstractFormula(Inference inference, AbstractFormula[] from) {
    this.inference = inference;
    this.from = from;
  }

  private void walkProof(Consumer<AbstractFormula> f, HashSet<AbstractFormula> visited) {
    if (visited.contains(this)) return;
    visited.add(this);
    for (var formula : from) formula.walkProof(f, visited);
    f.accept(this);
  }

  public final ArrayList<AbstractFormula> proof() {
    var r = new ArrayList<AbstractFormula>();
    walkProof(r::add, new HashSet<>());
    return r;
  }

  public abstract Object term();
}
