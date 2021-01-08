package prover;

public final class Formula extends AbstractFormula {
  private final Object term;

  protected Formula(Object term, Inference inference, AbstractFormula... from) {
    super(inference, from);
    this.term = term;
  }

  @Override
  public Object term() {
    return term;
  }
}
