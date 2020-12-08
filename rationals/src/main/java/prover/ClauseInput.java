package prover;

import java.util.List;

public final class ClauseInput extends Clause {
  private final String file;

  public ClauseInput(List<Term> negative, List<Term> positive, String name, String file) {
    super(negative, positive, name);
    this.file = file;
  }

  @Override
  public String file() {
    return file;
  }
}
