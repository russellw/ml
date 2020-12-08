package prover;

import java.util.List;

public class ClauseFrom extends Clause {
  private final Formula from;

  public ClauseFrom(List<Term> negative, List<Term> positive, Formula from) {
    super(negative, positive);
    this.from = from;
  }

  public ClauseFrom(Term[] literals, int negativeSize, Formula from) {
    super(literals, negativeSize);
    this.from = from;
  }

  @Override
  public final Formula[] from() {
    return new Formula[] {from};
  }
}
