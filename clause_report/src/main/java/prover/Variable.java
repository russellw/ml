package prover;

import java.util.HashMap;

public final class Variable {
  public final Object type;
  public static final HashMap<Variable, String> names = new HashMap<>();

  public Variable(Object type) {
    this.type = type;
  }

  @Override
  public String toString() {
    var name = names.get(this);
    if (name == null) {
      var i = names.size();
      name = i < 26 ? Character.toString('A' + i) : "Z" + (i - 25);
      names.put(this, name);
    }
    return name;
  }
}
