package prover;

import java.util.*;

import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

public class Clause extends Formula {
    public Term[] literals;
    public int negativeSize;

    public Clause(List<Term> negative, List<Term> positive) {
        init(negative, positive);
    }

    public Clause(Term[] literals, int negativeSize) {
        this.literals = literals;
        this.negativeSize = negativeSize;
    }

    public Clause(List<Term> negative, List<Term> positive, String name) {
        super(name);
        init(negative, positive);
    }

    private static void check(List<Term> literals) {
        for (var a : literals) {
            switch (a.op()) {
            case ALL:
            case AND:
            case EQV:
            case EXISTS:
            case NOT:
            case OR:
                throw new IllegalArgumentException(a.toString());
            }
        }
    }

    private void init(List<Term> negative, List<Term> positive) {
        check(negative);
        check(positive);

        // Simplify
        for (var i = 0; i < negative.size(); i++) {
            negative.set(i, negative.get(i).simplify());
        }
        for (var i = 0; i < positive.size(); i++) {
            positive.set(i, positive.get(i).simplify());
        }

        // Redundancy
        negative.removeIf(a -> a == Term.TRUE);
        positive.removeIf(a -> a == Term.FALSE);

        // Tautology?
        for (var a : negative) {
            if (a == Term.FALSE) {
                literals = new Term[] { Term.TRUE };
                return;
            }
        }
        for (var a : positive) {
            if (a == Term.TRUE) {
                literals = new Term[] { Term.TRUE };
                return;
            }
        }
        for (var a : negative) {
            for (var b : positive) {
                if (a.equals(b)) {
                    literals = new Term[] { Term.TRUE };
                    return;
                }
            }
        }

        // Literals
        literals = new Term[negative.size() + positive.size()];
        for (var i = 0; i < negative.size(); i++) {
            literals[i] = negative.get(i);
        }
        for (var i = 0; i < positive.size(); i++) {
            literals[negative.size() + i] = positive.get(i);
        }
        negativeSize = negative.size();
    }

    public final boolean isFalse() {
        return literals.length == 0;
    }

    public final boolean isTrue() {
        return (literals.length == 1) && (literals[0] == Term.TRUE);
    }

    public final Term[] negative() {
        return Arrays.copyOf(literals, negativeSize);
    }

    public static Set<Term> ops(Iterable<Clause> clauses) {
        var r = new HashSet<Term>();
        for (var c : clauses) {
            for (var a : c.literals) {
                a.getOps(r);
            }
        }
        return r;
    }

    public final Term[] positive() {
        return Arrays.copyOfRange(literals, negativeSize, literals.length);
    }

    public final int positiveSize() {
        return literals.length - negativeSize;
    }

    public final Term term() {
        if (isFalse()) {
            return Term.FALSE;
        }
        var r = new Term[1 + literals.length];
        r[0] = Term.OR;
        for (var i = 0; i < literals.length; i++) {
            var a = literals[i];
            if (i < negativeSize) {
                a = a.not();
            }
            r[i + 1] = a;
        }
        return Term.of(r).quantify();
    }

    public final void typeCheck() {
        for (var a : literals) {
            a.typeCheck(Type.BOOLEAN);
        }
    }

    @Override
    public void xml(XMLStreamWriter writer, int depth) throws XMLStreamException {
        Util.startElement(writer, depth, "Clause");
        writer.writeAttribute("class", getClass().getName());
        writer.writeAttribute("hash", Integer.toHexString(hashCode()));
        if (inference() != null) {
            writer.writeAttribute("inference", inference());
        }
        writer.writeAttribute("szs", szs().toString());
        writer.writeCharacters("\n");

        // Negative literals
        Util.startElement(writer, depth + 1, "negative");
        writer.writeCharacters("\n");
        for (var i = 0; i < negativeSize; i++) {
            literals[i].xml(writer, depth + 2);
        }
        Util.endElement(writer, depth + 1);
        writer.writeCharacters("\n");

        // Positive literals
        Util.startElement(writer, depth + 1, "positive");
        writer.writeCharacters("\n");
        for (var i = negativeSize; i < literals.length; i++) {
            literals[i].xml(writer, 2);
        }
        Util.endElement(writer, depth + 1);
        writer.writeCharacters("\n");

        // End
        Util.endElement(writer, depth);
        writer.writeCharacters("\n");
    }
}
