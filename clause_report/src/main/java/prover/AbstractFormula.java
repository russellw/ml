package prover;

public abstract class AbstractFormula {
  public Object name;
  public final Inference inference;
  public final AbstractFormula[] from;

  protected AbstractFormula(Inference inference, AbstractFormula[] from) {
    this.inference = inference;
    this.from = from;
  }

  public abstract Object term();
}
