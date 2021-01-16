package prover;

import java.util.HashMap;

public final class Bag<K> extends HashMap<K, Integer> {
  public void add(K key) {
    var n = get(key);
    if (n == null) n = 0;
    put(key, n + 1);
  }
}
