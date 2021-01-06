package prover;

public abstract class AbstractFormula {
  public Object name;
  public final AbstractFormula[] from;

  protected AbstractFormula(AbstractFormula[] from) {
    this.from = from;
  }

  public abstract Object term();
}
