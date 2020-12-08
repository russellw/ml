package prover;

import java.util.Arrays;

// Set of nonnegative integers
// More efficient than HashSet if the range is reasonably compact
public class SetInt {
  private boolean[] data = new boolean[0];

  public void add(int value) {
    assert value >= 0;
    if (data.length <= value) {
      data = Arrays.copyOf(data, Math.max(value + 1, data.length * 2));
    }
    data[value] = true;
  }

  public int size() {
    var n = 0;
    for (var value : data) {
      if (value) {
        n++;
      }
    }
    return n;
  }
}
