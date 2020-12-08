package prover;

import java.util.*;
import java.util.concurrent.TimeoutException;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

// Open problem:
// https://stackoverflow.com/questions/54043747/clause-subsumption-algorithm
public final class Subsumption {
  public static int limitOpsConsidered = 651;
  public static int limitOpsUsed = 76;
  public static int limitSteps = 77241;

  // Categories of per-operator features
  // This order is chosen to place the least informative features first
  // We use a search trie in an unusual way
  // that causes this order to have no effect on number of operations performed
  // but makes the trie narrower at the top, which saves memory
  private static final int NEGATIVE_OP_MAX_DEPTH = 0;
  private static final int NEGATIVE_OP_COUNT = 1;
  private static final int POSITIVE_OP_MAX_DEPTH = 2;
  private static final int POSITIVE_OP_COUNT = 3;

  // Operators used for features
  private Term[] ops;

  // and corresponding array indexes
  private Map<Term, Integer> indexes;

  // Index for fast filtering algorithm described in
  // Simple and Efficient Clause Subsumption with Feature Vector Indexing
  // by Stephan Schulz
  private WeakTrie<Clause> trie = new WeakTrie<>();

  // Time limit
  private int steps;

  // Which clauses have been subsumed?
  private WeakHashMap<Clause, Boolean> subsumed = new WeakHashMap<>();

  // Statistics
  public Histogram histogram =
      new Histogram() {
        @Override
        public String keyHeader() {
          return "Operator";
        }

        @Override
        public String keyString(Object key) {
          var s = TptpPrinter.called((Term) key);
          var limit = 40;
          if (s.length() > limit) {
            s = s.substring(0, limit);
          }
          s = s.replace("<", "&lt;");
          return String.format("<code>%s</code>", s);
        }

        @Override
        public String valueHeader() {
          return "Initial range";
        }
      };
  public long attempted;
  public long succeeded;

  public Subsumption(Iterable<Clause> clauses) {

    // What operators are used in these clauses?
    // That will be the basis of features for filtering subsumption candidates
    ops = Clause.ops(clauses).toArray(new Term[0]);
    if (ops.length > limitOpsConsidered) {
      ops = Arrays.copyOf(ops, limitOpsConsidered);
    }

    // and corresponding array indexes
    indexes = new HashMap<>(ops.length);
    for (var term : ops) {
      indexes.put(term, indexes.size());
    }

    // For each operator, what range of values does it generate for its features?
    // That indicates how informative that operator is
    var r = new OpRange[ops.length];
    for (var i = 0; i < r.length; i++) {
      r[i] = new OpRange(ops[i]);
    }
    for (var c : clauses) {
      var features = features(c);
      for (var i = 0; i < r.length; i++) {
        var operator = r[i];
        for (var j = 0; j < 4; j++) {
          operator.sets[j].add(features[2 + j * r.length + i]);
        }
      }
    }

    // Rank operators from least to most informative
    Arrays.sort(r, Comparator.comparingInt(OpRange::range));

    // There is a limit to how many operators it's worth spending resources filtering by
    var n = Math.min(r.length, limitOpsUsed);

    // Take the most informative operators
    ops = new Term[n];

    // But the selected ones will then be ranked least informative first
    // This means the trie uses less memory
    for (var i = 0; i < ops.length; i++) {
      var operator = r[r.length - n + i];
      ops[i] = operator.term;
      histogram.map.put(operator.term, new ValueSubsumption(operator.ints()));
    }

    // and corresponding array indexes
    indexes = new HashMap<>(ops.length);
    for (var term : ops) {
      indexes.put(term, indexes.size());
    }
  }

  public boolean add(Clause c) {

    // Forward subsumption
    if (trie.findLessEqual(
            features(c),
            d -> {
              if (subsumed(d)) {
                return false;
              }
              return subsumes(d, c);
            })
        != null) {
      return false;
    }

    // Backward subsumption
    trie.forGreaterEqual(
        features(c),
        d -> {
          if (subsumed(d)) {
            return;
          }
          if (subsumes(c, d)) {
            subsumed.put(d, true);
          }
        });

    // Add to index
    trie.add(features(c), c);
    return true;
  }

  public void addQuick(Clause c) {
    trie.add(features(c), c);
  }

  private static DeterministicMatches deterministicMatches(
      Term[] c, Term[] d, Map<Variable, Term> map) {

    // Top-level occurrences of each symbol
    var cs = uniques(c);
    var ds = uniques(d);

    // Remember which literals deterministically matched
    var cmatched = new boolean[c.length];
    var dmatched = new boolean[d.length];

    // Symbols that only occur once in c
    for (var key : cs.keySet()) {
      var ci = cs.get(key);
      if (ci < 0) {
        continue;
      }
      assert !cmatched[ci];

      // And only once in d
      var di = ds.get(key);
      if ((di == null) || (di < 0)) {
        continue;
      }
      assert !dmatched[di];

      // Are matched deterministically if at all
      if (!c[ci].match(d[di], map)) {
        return null;
      }
      cmatched[ci] = true;
      dmatched[di] = true;
    }

    // How many matched?
    var matched = 0;
    for (var m : cmatched) {
      if (m) {
        matched++;
      }
    }

    // Unmatched c literals
    var c1 = new Term[c.length - matched];
    var j = 0;
    for (var i = 0; i < c.length; i++) {
      if (!cmatched[i]) {
        c1[j++] = c[i];
      }
    }
    assert j == c1.length;

    // Unmatched d literals
    var d1 = new Term[d.length - matched];
    j = 0;
    for (var i = 0; i < d.length; i++) {
      if (!dmatched[i]) {
        d1[j++] = d[i];
      }
    }
    assert j == d1.length;

    // Return multiple values as object
    return new DeterministicMatches(c1, d1, map);
  }

  private int[] features(Clause c) {
    var r = new int[2 + 4 * ops.length];

    // Global features:
    // negative literal count
    r[0] = c.negativeSize;

    // positive literal count
    r[1] = c.positiveSize();

    // Per-operator features:
    // negative literals: operator max depth
    // negative literals: operator count
    for (var i = 0; i < c.negativeSize; i++) {
      var a = c.literals[i];
      a.walk(
          (b, depth) -> {
            if (b.size() > 0) {
              b = b.get(0);
            }
            var j = indexes.get(b);
            if (j == null) {
              return;
            }

            // Depth 0 (top level) is not distinct from depth blank (nonexistent)
            // This is okay because operator count will already distinguish between these cases
            r[2 + NEGATIVE_OP_MAX_DEPTH * ops.length + j] =
                Math.max(r[2 + NEGATIVE_OP_MAX_DEPTH * ops.length + j], depth);
            r[2 + NEGATIVE_OP_COUNT * ops.length + j]++;
          });
    }

    // positive literals: operator max depth
    // positive literals: operator count
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var a = c.literals[i];
      a.walk(
          (b, depth) -> {
            if (b.size() > 0) {
              b = b.get(0);
            }
            var j = indexes.get(b);
            if (j == null) {
              return;
            }
            r[2 + POSITIVE_OP_MAX_DEPTH * ops.length + j] =
                Math.max(r[2 + POSITIVE_OP_MAX_DEPTH * ops.length + j], depth);
            r[2 + POSITIVE_OP_COUNT * ops.length + j]++;
          });
    }
    return r;
  }

  private Map<Variable, Term> search(
      Term[] c, Term[] c2, Term[] d, Term[] d2, Map<Variable, Term> map) throws TimeoutException {
    if (steps == 0) {
      throw new TimeoutException();
    }
    steps--;

    // Matched everything in one polarity
    if (c.length == 0) {

      // Matched everything in the other polarity
      if (c2 == null) {
        return map;
      }

      // Try the other polarity
      return search(c2, null, d2, null, map);
    }

    // Try matching literals
    for (var ci = 0; ci < c.length; ci++) {
      Term[] c1 = null;
      var ce = Equation.of(c[ci]);
      for (var di = 0; di < d.length; di++) {
        Term[] d1 = null;
        var de = Equation.of(d[di]);

        // Search means preserve the original map
        // in case the search fails
        // and need to backtrack
        Map<Variable, Term> m;

        // Try orienting equation one way
        m = new HashMap<>(map);
        if (ce.left.match(de.left, m) && ce.right.match(de.right, m)) {
          if (c1 == null) {
            c1 = Term.remove(c, ci);
          }
          d1 = Term.remove(d, di);
          m = search(c1, c2, d1, d2, m);
          if (m != null) {
            return m;
          }
        }

        // And the other way
        m = new HashMap<>(map);
        if (ce.left.match(de.right, m) && ce.right.match(de.left, m)) {
          if (c1 == null) {
            c1 = Term.remove(c, ci);
          }
          if (d1 == null) {
            d1 = Term.remove(d, di);
          }
          m = search(c1, c2, d1, d2, m);
          if (m != null) {
            return m;
          }
        }
      }
    }

    // No match
    return null;
  }

  public boolean subsumed(Clause c) {
    return subsumed.containsKey(c);
  }

  public boolean subsumes(Clause c, Clause d) {
    attempted++;

    // Negative and positive literals must subsume separately
    var c1 = c.negative();
    var c2 = c.positive();
    var d1 = d.negative();
    var d2 = d.positive();

    // Fewer literals typically fail faster
    if (c2.length < c1.length) {

      // Swap negative and positive
      var ct = c1;
      c1 = c2;
      c2 = ct;

      // And in the other clause
      var dt = d1;
      d1 = d2;
      d2 = dt;
    }

    // Worst-case time is exponential
    // so give up if taking too long
    steps = limitSteps;
    try {
      Map<Variable, Term> map = new HashMap<>();

      // Negative literals (unless swapped)
      var dm = deterministicMatches(c1, d1, map);
      if (dm == null) {
        return false;
      }
      map = dm.map;
      c1 = dm.c;
      d1 = dm.d;

      // Positive literals (unless swapped)
      dm = deterministicMatches(c2, d2, map);
      if (dm == null) {
        return false;
      }
      map = dm.map;
      c2 = dm.c;
      d2 = dm.d;

      // Search for nondeterministic matches
      map = search(c1, c2, d1, d2, map);
      if (map != null) {
        succeeded++;
      }
      return map != null;
    } catch (TimeoutException e) {
      return false;
    }
  }

  private static void uniqueCandidate(Map<Term, Integer> map, Term key, int value) {
    var old = map.get(key);
    if (old == null) {
      map.put(key, value);
      return;
    }
    if (old >= 0) {
      map.put(key, -1);
    }
  }

  private static Map<Term, Integer> uniques(Term[] c) {
    var map = new HashMap<Term, Integer>();
    for (var i = 0; i < c.length; i++) {
      var a = c[i];
      if (a.size() == 0) {
        uniqueCandidate(map, a, i);
        continue;
      }
      if (a.op() == Op.EQ) {
        continue;
      }
      uniqueCandidate(map, a.get(0), i);
    }
    return map;
  }

  public void xml() {
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

  public void xml(XMLStreamWriter writer, int depth) throws XMLStreamException {
    Util.startElement(writer, depth, "subsumption");
    writer.writeCharacters("\n");

    // Operators
    Util.startElement(writer, depth + 1, "ops");
    writer.writeCharacters("\n");
    for (var op : ops) {
      Util.text(writer, depth + 2, op.toString());
      writer.writeCharacters("\n");
    }
    Util.endElement(writer, depth + 1);
    writer.writeCharacters("\n");

    // Indexes
    Util.startElement(writer, depth + 1, "indexes");
    writer.writeCharacters("\n");
    for (var op : indexes.keySet()) {
      Util.text(writer, depth + 2, op + ": " + indexes.get(op));
      writer.writeCharacters("\n");
    }
    Util.endElement(writer, depth + 1);
    writer.writeCharacters("\n");

    // Subsumed
    Util.startElement(writer, depth + 1, "subsumed");
    writer.writeCharacters("\n");
    for (var c : subsumed.keySet()) {
      Util.text(writer, depth + 2, c.toString());
      writer.writeCharacters("\n");
    }
    Util.endElement(writer, depth + 1);
    writer.writeCharacters("\n");

    // End
    Util.endElement(writer, depth);
    writer.writeCharacters("\n");
  }

  private static final class DeterministicMatches {
    final Term[] c;
    final Term[] d;
    final Map<Variable, Term> map;

    DeterministicMatches(Term[] c, Term[] d, Map<Variable, Term> map) {
      this.c = c;
      this.d = d;
      this.map = map;
    }
  }

  private static final class OpRange {
    final Term term;
    final SetInt[] sets = new SetInt[4];

    OpRange(Term term) {
      this.term = term;
      for (var i = 0; i < 4; i++) {
        sets[i] = new SetInt();
      }
    }

    int[] ints() {
      var r = new int[sets.length];
      for (var i = 0; i < r.length; i++) {
        r[i] = sets[i].size();
      }
      return r;
    }

    int range() {
      var n = 0;
      for (var set : sets) {
        n += set.size();
      }
      return n;
    }
  }

  private static final class ValueSubsumption extends HistogramValue {
    private final int[] range;

    public ValueSubsumption(int[] range) {
      this.range = range;
    }

    @Override
    public String toString() {
      var sb = new StringBuilder();
      for (var n : range) {
        if (sb.length() > 0) {
          sb.append(',');
        }
        sb.append(n);
      }
      return sb.toString();
    }

    @Override
    public int val() {
      var sum = 0;
      for (var n : range) {
        sum += n;
      }
      return sum;
    }
  }
}
