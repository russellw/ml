package prover;

public abstract class HistogramValue {
  @Override
  public String toString() {
    return Integer.toString(val());
  }

  public abstract int val();
}
