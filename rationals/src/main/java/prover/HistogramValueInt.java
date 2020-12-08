package prover;

public final class HistogramValueInt extends HistogramValue {
  private final int val;

  public HistogramValueInt(int val) {
    this.val = val;
  }

  @Override
  public int val() {
    return val;
  }
}
