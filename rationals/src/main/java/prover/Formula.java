package prover;

import java.util.*;
import java.util.function.Consumer;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

public abstract class Formula {
  public static long nextId;
  private String name;
  private long id = -1;

  public Formula() {}

  public Formula(String name) {
    this.name = name;
  }

  public String file() {
    return null;
  }

  public Formula[] from() {
    return new Formula[0];
  }

  public static Set<Function> functions(Formula[] formulas) {
    var r = new HashSet<Function>();
    for (var formula : formulas) {
      formula.term().getFuncs(r);
    }
    return r;
  }

  public static Set<Function> functions(Iterable<? extends Formula> formulas) {
    var r = new HashSet<Function>();
    for (var formula : formulas) {
      formula.term().getFuncs(r);
    }
    return r;
  }

  public String inference() {
    return null;
  }

  public final List<Formula> proof() {
    var r = new ArrayList<Formula>();
    walkProof(r::add, new HashSet<>());
    return r;
  }

  public static void reserveId(String name) {
    if (Util.isDigits(name)) {
      nextId = Math.max(nextId, Long.parseLong(name) + 1);
    }
  }

  public void setId() {
    if (name == null) {
      id = nextId++;
    }
  }

  public final SZS szs() {
    var from = from();
    switch (from.length) {
      case 0:

        // Presumably input data
        return SZS.LogicalData;
      case 1:
        {
          var a = from[0].term();
          var b = term();

          // Presumably variable renaming
          if (a.isomorphic(b, new HashMap<>())) {
            return SZS.Equivalent;
          }

          // Presumably negated conjecture
          if (a.not().isomorphic(b, new HashMap<>())) {
            return SZS.CounterEquivalent;
          }
          break;
        }
    }

    // If a formula introduces new symbols, then it is only equisatisfiable
    // This happens during subformula renaming in CNF conversion
    var fromFuncs = functions(from);
    for (var a : term().functions()) {
      if (!fromFuncs.contains(a)) {
        return SZS.EquiSatisfiable;
      }
    }

    // Straightforward derivation
    return SZS.Theorem;
  }

  public abstract Term term();

  @Override
  public final String toString() {
    if (name != null) {
      if (Util.isDigits(name)) {
        return name;
      }
      if (!TptpPrinter.weird(name)) {
        return name;
      }
      return Util.quote('\'', name);
    }
    if (id >= 0) {
      return Long.toString(id);
    }
    throw new IllegalStateException();
  }

  private void walkProof(Consumer<Formula> f, Set<Formula> visited) {
    if (visited.contains(this)) {
      return;
    }
    visited.add(this);
    for (var formula : from()) {
      formula.walkProof(f, visited);
    }
    f.accept(this);
  }

  public final void xml() {
    try {
      var writer = XMLOutputFactory.newFactory().createXMLStreamWriter(System.out);
      writer.writeStartDocument();
      writer.writeCharacters("\n");
      xml(writer, 0);
      writer.writeEndDocument();
      writer.close();
    } catch (XMLStreamException e) {
      throw new RuntimeException(e);
    }
  }

  public static void xml(Iterable<? extends Formula> formulas) {
    try {
      var writer = XMLOutputFactory.newFactory().createXMLStreamWriter(System.out);
      writer.writeStartDocument();
      writer.writeCharacters("\n");
      for (var formula : formulas) {
        formula.xml(writer, 0);
      }
      writer.writeEndDocument();
      writer.close();
    } catch (XMLStreamException e) {
      throw new RuntimeException(e);
    }
  }

  public void xml(XMLStreamWriter writer, int depth) throws XMLStreamException {
    Util.startElement(writer, depth, "Clause");
    writer.writeAttribute("class", getClass().getName());
    writer.writeAttribute("hash", Integer.toHexString(hashCode()));
    if (inference() != null) {
      writer.writeAttribute("inference", inference());
    }
    writer.writeAttribute("szs", szs().toString());
    writer.writeCharacters("\n");

    // Term
    term().xml(writer, depth + 1);

    // End
    Util.endElement(writer, depth);
    writer.writeCharacters("\n");
  }
}
