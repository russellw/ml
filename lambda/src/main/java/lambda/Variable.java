package lambda;

public final class Variable {
  public final Object type;
  public final Object value;

  public Variable(Object type) {
    this.type = type;
    value = null;
  }

  public Variable(Object type, Object value) {
    this.type = type;
    this.value = value;
  }
}
