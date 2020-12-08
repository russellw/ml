package prover;

public class FormulaTermFrom extends FormulaTerm {
  private final Formula from;

  public FormulaTermFrom(Term term, Formula from) {
    super(term);
    this.from = from;
  }

  @Override
  public Formula[] from() {
    return new Formula[] {from};
  }
}
