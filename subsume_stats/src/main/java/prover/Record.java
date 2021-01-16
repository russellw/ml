package prover;

import java.util.Collection;

public class Record {
  public int activeLive;
  public int activeTotal;
  public int passiveLive;
  public int passiveTotal;

  Record(Collection<Clause> active, Collection<Clause> passive) {
    for (var c : active) if (!c.subsumed) activeLive++;
    activeTotal = active.size();
    for (var c : passive) if (!c.subsumed) passiveLive++;
    passiveTotal = passive.size();
  }
}
