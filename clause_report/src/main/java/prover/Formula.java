package prover;

public final class Formula extends AbstractFormula {
  private final Object term;

  protected Formula(Object term, AbstractFormula... from) {
    super(from);
    this.term = term;
  }

  @Override
  public Object term() {
    return term;
  }
}
