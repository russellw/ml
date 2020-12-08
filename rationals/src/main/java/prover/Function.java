package prover;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

public final class Function extends Term {
  public static long nextId;
  private Type type;
  private String name;

  public Function(Type type, String name) {
    this.type = type;
    setName(name);
  }

  public Function(Type returnType, String name, Collection<? extends Term> args) {
    this.type = params(returnType, args);
    setName(name);
  }

  public int arity() {
    if (type.size() == 0) {
      return 0;
    }
    return type.size() - 1;
  }

  public Term call(Collection<? extends Term> args) {
    var args1 = new Term[1 + args.size()];
    args1[0] = this;
    var i = 1;
    for (var a : args) {
      args1[i++] = a;
    }
    return of(args1);
  }

  public Term call(Term... args) {
    var args1 = new Term[1 + args.length];
    args1[0] = this;
    var i = 1;
    for (var a : args) {
      args1[i++] = a;
    }
    return of(args1);
  }

  @Override
  public void getFuncs(Set<Function> r) {
    r.add(this);
  }

  @Override
  public void getOps(Set<Term> r) {
    r.add(this);
  }

  @Override
  public boolean isConstant() {
    return false;
  }

  public boolean isPredicate() {
    if (type == Type.BOOLEAN) {
      return true;
    }
    if (type.size() > 0) {
      return type.get(0) == Type.BOOLEAN;
    }
    return false;
  }

  private static Type params(Type returnType, Collection<? extends Term> args) {
    var params = new Type[1 + args.size()];
    params[0] = returnType;
    var i = 1;
    for (var a : args) {
      params[i++] = a.type();
    }
    return Type.of(params);
  }

  public static void reserveId(String name) {
    if (!name.startsWith("sK")) {
      return;
    }
    for (int i = 2; i < name.length(); i++) {
      if (!Character.isDigit(name.charAt(i))) {
        return;
      }
    }
    nextId = Math.max(nextId, Long.parseLong(name.substring(2)) + 1);
  }

  public void setName(String name) {
    if (name == null) {
      name = "sK" + nextId++;
    }
    this.name = name;
  }

  @Override
  public Tag tag() {
    return Tag.FUNC;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public Type type() {
    return type;
  }

  @Override
  public void typeAssign(Map<TypeVariable, Type> map) {
    type = type.replaceVars(map);
    switch (type.kind()) {
      case FUNCTION:
        {
          var params = new Type[type.size()];
          for (var i = 0; i < params.length; i++) {
            var param = type.get(i);
            if (param instanceof TypeVariable) {
              param = Type.INDIVIDUAL;
            }
            params[i] = param;
          }
          type = Type.of(params);
          break;
        }
      case VARIABLE:
        type = Type.INDIVIDUAL;
        break;
    }
  }
}
