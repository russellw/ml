package prover;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import java.text.NumberFormat;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import java.util.*;
import java.util.List;

public final class Problem {
    private static List<Summary> summaries = new ArrayList<>();
    private static PrintWriter writer;
    public Formula conjecture;
    public List<String> header = new ArrayList<>();
    public SZS expected;
    public SZS result;
    private long nextIdFunc;
    private long nextIdFormula;
    public final long timeBegin = System.currentTimeMillis();
    public long timeEnd;
    public long timeReadEnd;
    public final List<Clause> clauses;
    public final List<Formula> formulas = new ArrayList<>();
    private CNF cnf;
    private List<InputFile> includeStack = new ArrayList<>();
    private InputFile root;
    public Superposition superposition;

    public Problem() {
        clauses = new ArrayList<>();
    }

    public Problem(List<Clause> clauses) {
        this.clauses = clauses;
    }

    private static void args(Term term) {
        writer.print('(');
        for (var i = 1; i < term.size(); i++) {
            if (i > 1) {
                writer.print(", ");
            }
            print(term.get(i), null);
        }
        writer.print(')');
    }

    public void enter(Path path) {
        var file = new InputFile(path);
        if (root == null) {
            root = file;
        }
        if (includeStack.size() > 0) {
            includeStack.get(includeStack.size() - 1).includes.add(file);
        }
        includeStack.add(file);
    }

    private static void infix(Term term, String op) {
        for (var i = 1; i < term.size(); i++) {
            if (i > 1) {
                writer.print(op);
            }
            print(term.get(i), term);
        }
    }

    public static Language language(Path path) throws IOException {
        if (Main.language != null) {
            return Main.language;
        }
        switch (Util.extension(path.toString())) {
        case "ax":
        case "p":
            return Language.TPTP;
        case "cnf":
            return Language.DIMACS;
        }
        throw new IOException(path + ": language not specified");
    }

    public void leave() {
        includeStack.remove(includeStack.size() - 1);
    }

    private static String mathLetter(char c) {
        if (Character.isUpperCase(c)) {
            return String.format("&#x%x", 0x1d434 + c - 'A');
        }
        if (Character.isLowerCase(c)) {
            return String.format("&#x%x", 0x1d44e + c - 'a');
        }
        throw new IllegalArgumentException(Character.toString(c));
    }

    private static void open(String file) throws IOException {
        file = Util.removeExtension(file);
        writer = new PrintWriter("reports/" + file + ".html", StandardCharsets.UTF_8);
        writer.println("<!DOCTYPE html>");
        writer.println("<html lang=\"en\">");
        writer.println("<meta charset=\"utf-8\"/>");
        writer.println("<style>");
        writer.println("h1 {");
        writer.println("font-size: 150%;");
        writer.println("}");
        writer.println("h2 {");
        writer.println("font-size: 125%;");
        writer.println("}");
        writer.println("caption {");
        writer.println("text-align: left;");
        writer.println("white-space: nowrap;");
        writer.println("}");
        writer.println("table.bordered, th.bordered, td.bordered {");
        writer.println("border: 1px solid;");
        writer.println("border-collapse: collapse;");
        writer.println("padding: 5px;");
        writer.println("}");
        writer.println("table.padded, th.padded, td.padded {");
        writer.println("padding: 3px;");
        writer.println("}");
        writer.println("td.fixed {");
        writer.println("white-space: nowrap;");
        writer.println("}");
        writer.println("td.bar {");
        writer.println("width: 100%");
        writer.println("}");
        writer.println("</style>");
        writer.printf("<title>%s</title>\n", file);
    }

    public void print(String file) throws IOException {
        open(file);
        var numberFormat = NumberFormat.getInstance();

        // Contents
        writer.println("<h1 id=\"Contents\">Contents</h1>");
        writer.println("<ul>");
        writer.println("<li><a href=\"#Contents\">Contents</a>");
        writer.println("<li><a href=\"#Input-files\">Input files</a>");
        if (!header.isEmpty()) {
            writer.println("<li><a href=\"#Problem-header\">Problem header</a>");
        }
        writer.println("<li><a href=\"#Clauses\">Clauses</a>");
        writer.println("<li><a href=\"#Subsumption\">Subsumption</a>");
        writer.println("<li><a href=\"#Result\">Result</a>");
        if (superposition.conclusion != null) {
            writer.println("<li><a href=\"#Proof\">Proof</a>");
        }
        writer.println("<li><a href=\"#Memory\">Memory</a>");
        writer.println("<li><a href=\"#Time\">Time</a>");
        if (Main.version() != null) {
            writer.println("<li><a href=\"#Version\">Version</a>");
        }
        writer.println("</ul>");

        // Input files
        writer.println("<h1 id=\"Input-files\">Input files</h1>");
        writer.println("<ul>");
        writer.println("<li>");
        root.print();
        writer.println("</ul>");
        if ((nextIdFunc | nextIdFormula) != 0) {
            writer.println("<table class=\"bordered\">");
            writer.println("<caption>Max input ID");
            if (nextIdFunc != 0) {
                writer.println("<tr>");
                writer.println("<td class=\"bordered\">Function");
                writer.println("<td class=\"bordered\">sK" + (nextIdFunc - 1));
            }
            if (nextIdFormula != 0) {
                writer.println("<tr>");
                writer.println("<td class=\"bordered\">Formula");
                writer.println("<td class=\"bordered\">" + (nextIdFormula - 1));
            }
            writer.println("</table>");
        }

        // Problem header
        if (!header.isEmpty()) {
            if (header.get(header.size() - 1).isEmpty()) {
                header.remove(header.size() - 1);
            }
            writer.println("<h1 id=\"Problem-header\">Problem header</h1>");
            writer.println("<pre>");
            for (var s : header) {
                wrap(s);
            }
            writer.println("</pre>");
        }

        // Clauses
        writer.println("<h1 id=\"Clauses\">Clauses</h1>");
        if (cnf != null) {
            writer.println("<table>");
            writer.println("<caption>FOF &rarr; CNF expansion factors");
            cnf.histogram.print(writer);
            writer.println("</table>");
            writer.println("<br>");
        }
        writer.println("<table class=\"bordered\">");
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Initial");
        writer.println("<td class=\"bordered\">" + ((cnf == null)
                                                    ? "Input"
                                                    : "CNF conversion"));
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(clauses.size()));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Generated");
        writer.println("<td class=\"bordered\">Resolution");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.generatedResolution));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">");
        writer.println("<td class=\"bordered\">Factoring");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.generatedFactoring));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">");
        writer.println("<td class=\"bordered\">Superposition");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.generatedSuperposition));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Discarded");
        writer.println("<td class=\"bordered\">Tautologies");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(superposition.tautologies));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">");
        writer.println("<td class=\"bordered\">Subsumed forward");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.subsumedForward));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">");
        writer.println("<td class=\"bordered\">Subsumed backward");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.subsumedBackward));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Final");
        writer.println("<td class=\"bordered\">Processed");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.processed.size()));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">");
        writer.println("<td class=\"bordered\">Unprocessed");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.unprocessed.size()));
        writer.println("</table>");
        var clausesGenerated = superposition.generatedResolution + superposition.generatedFactoring
                               + superposition.generatedSuperposition;
        var clausesDiscarded = superposition.tautologies + superposition.subsumedForward + superposition.subsumedBackward;
        var clausesFinal = superposition.processed.size() + superposition.unprocessed.size();
        if (superposition.processed.size() > 0) {
            writer.println("<h2 id=\"Processed\">Processed</h2>");
            printSample(superposition.processed);
        }
        if (superposition.unprocessed.size() > 0) {
            writer.println("<h2 id=\"Unprocessed\">Unprocessed</h2>");

            // Priority queue iterator does not follow queue order
            // so need to make an ordered copy
            var unprocessed = new ArrayList<Clause>();
            while (superposition.unprocessed.size() > 0) {
                unprocessed.add(superposition.unprocessed.poll());
            }
            printSample(unprocessed);
        }

        // Subsumption
        writer.println("<h1 id=\"Subsumption\">Subsumption</h1>");
        writer.println("<table class=\"bordered\">");
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Attempted");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.subsumption.attempted));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Succeeded");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(superposition.subsumption.succeeded));
        writer.println("</table>");
        writer.println("<br>");
        writer.println("<table>");
        writer.printf("<caption>%d operators used for feature vector indexing: ", superposition.subsumption.histogram.map.size());
        writer.printf("%s<sub>%s</sub>(%s<sup>-</sup>), |%s<sup>-</sup>|<sub>%s</sub>, ",
                      mathLetter('d'),
                      mathLetter('f'),
                      mathLetter('C'),
                      mathLetter('C'),
                      mathLetter('f'));
        writer.printf("%s<sub>%s</sub>(%s<sup>+</sup>), |%s<sup>+</sup>|<sub>%s</sub>",
                      mathLetter('d'),
                      mathLetter('f'),
                      mathLetter('C'),
                      mathLetter('C'),
                      mathLetter('f'));
        writer.println();
        superposition.subsumption.histogram.print(writer);
        writer.println("</table>");

        // Result
        writer.println("<h1 id=\"Result\">Result</h1>");
        writer.println("<table class=\"bordered\">");
        if (expected != null) {
            writer.println("<tr>");
            writer.println("<td class=\"bordered\">Expected");
            writer.println("<td class=\"bordered\">" + expected);
        }
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Result");
        writer.printf("<td class=\"bordered\"><b>%s</b>\n", result);
        writer.println("</table>");
        if (!result.compatible(expected)) {
            writer.println("<p><b>***ERROR***</b>");
        }

        // Proof
        if (superposition.conclusion != null) {
            writer.println("<h1 id=\"Proof\">Proof</h1>");
            writer.println("<code>");
            for (var formula : superposition.conclusion.proof()) {
                var sublanguage = TptpPrinter.sublanguage(formula);
                var term = formula.term();
                if ("cnf".equals(sublanguage)) {
                    term = term.unquantify();
                }
                TptpPrinter.nameVars(term);
                writer.printf("%s(%s, %s, ", sublanguage, formula, TptpPrinter.role(formula));
                writer.print("<span style=\"color: #00c\">");
                print(term, null);
                writer.print("</span>");
                writer.print(", " + source(formula));
                writer.println(").<br>");
            }
            writer.println("</code>");
        }

        // Memory
        var runtime = Runtime.getRuntime();
        writer.println("<h1 id=\"Memory\">Memory</h1>");
        writer.println("<table class=\"bordered\">");
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Current");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">"
                       + numberFormat.format(runtime.totalMemory() - runtime.freeMemory()));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Free");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">+ " + numberFormat.format(runtime.freeMemory()));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Total");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">= " + numberFormat.format(runtime.totalMemory()));
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Max");
        writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(runtime.maxMemory()));
        writer.println("</table>");

        // Time
        writer.println("<h1 id=\"Time\">Time</h1>");
        writer.println("<table class=\"bordered\">");
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Read input");
        writer.printf("<td class=\"bordered\"; style=\"text-align: right\">%.3f\n", (timeReadEnd - timeBegin) / 1000.0);
        var timePrepBegin = timeReadEnd;
        if (cnf != null) {
            writer.println("<tr>");
            writer.println("<td class=\"bordered\">Convert to CNF");
            writer.printf("<td class=\"bordered\"; style=\"text-align: right\">%.3f\n", (cnf.timeEnd - timeReadEnd) / 1000.0);
            timePrepBegin = cnf.timeEnd;
        }
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Preparation");
        writer.printf("<td class=\"bordered\"; style=\"text-align: right\">%.3f\n",
                      (superposition.timePrepEnd - timePrepBegin) / 1000.0);
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Calculation");
        writer.printf("<td class=\"bordered\"; style=\"text-align: right\">+ %.3f\n",
                      (timeEnd - superposition.timePrepEnd) / 1000.0);
        writer.println("<tr>");
        writer.println("<td class=\"bordered\">Total");
        writer.printf("<td class=\"bordered\"; style=\"text-align: right\">= %.3f\n", (timeEnd - timeBegin) / 1000.0);
        writer.println("</table>");
        writer.println("<p>" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy, HH:mm:ss")));

        // Version
        if (Main.version() != null) {
            writer.println("<h1 id=\"Version\">Version</h1>");
            writer.println("<p>Prover " + Main.version());
        }

        // End
        writer.close();

        // Summary record
        var functions = Formula.functions(clauses);
        summaries.add(new Summary(Util.removeExtension(file),
                                  Util.count(functions, Function::isPredicate),
                                  functions.size() - Util.count(functions, Function::isPredicate),
                                  formulas.size(),
                                  clauses.size(),
                                  clausesGenerated,
                                  clausesDiscarded,
                                  clausesFinal,
                                  expected,
                                  result,
                                  timeEnd - timeBegin));
    }

    private static void print(Type type) {
        switch (type.kind()) {
        case BOOLEAN:
            writer.print("$o");
            return;
        case INDIVIDUAL:
            writer.print("$i");
            return;
        case INTEGER:
            writer.print("$int");
            return;
        case RATIONAL:
            writer.print("$rat");
            return;
        case REAL:
            writer.print("$real");
            return;
        default:
            throw new IllegalArgumentException(type.toString());
        }
    }

    private static void print(Term term, Term parent) {
        switch (term.tag()) {
        case CONST_FALSE:
            writer.print("$false");
            return;
        case CONST_TRUE:
            writer.print("$true");
            return;
        case FUNC: {
            var name = term.toString();
            if (!Character.isLowerCase(name.charAt(0)) || TptpPrinter.weird(name)) {
                name = Util.quote('\'', name);
            }
            writer.print(name);
            return;
        }
        case LIST:
            break;
        default:
            writer.print(term.toString());
            return;
        }
        if (TptpPrinter.needParens(term, parent)) {
            writer.print('(');
        }
        switch (term.op()) {
        case ALL:
            writer.print('!');
            quant(term);
            break;
        case AND:
            infix(term, " & ");
            break;
        case EQ:
            infix(term, " = ");
            break;
        case EQV:
            infix(term, " <=> ");
            break;
        case EXISTS:
            writer.print('?');
            quant(term);
            break;
        case NOT:
            if (term.get(0).op() == Op.EQ) {
                infix(term.get(1), " != ");
                break;
            }
            writer.print('~');
            print(term.get(1), term);
            break;
        case OR:
            infix(term, " | ");
            break;
        default:
            writer.print(TptpPrinter.called(term.get(0)).replace("<", "&lt;"));
            args(term);
            break;
        }
        if (TptpPrinter.needParens(term, parent)) {
            writer.print(')');
        }
    }

    private void printSample(Clause clause) {
        var sublanguage = TptpPrinter.sublanguage(clause);
        var term = clause.term();
        if ("cnf".equals(sublanguage)) {
            term = term.unquantify();
        }
        TptpPrinter.nameVars(term);
        var fade = (superposition.conclusion == null) && superposition.subsumption.subsumed(clause);
        if (fade) {
            writer.print("<span style=\"color: #888\">");
        }
        writer.printf("%s(%s, %s, ", sublanguage, clause, TptpPrinter.role(clause));
        writer.printf("<span style=\"color: #%s\">",
                      fade
                      ? "88e"
                      : "00c");
        print(term, null);
        writer.print("</span>).");
        if (fade) {
            writer.print("</span>");
        }
        writer.println("<br>");
    }

    private void printSample(List<Clause> clauses) {
        writer.print("<p>");
        writer.print(clauses.size() + " clauses");
        if (superposition.conclusion == null) {
            var subsumed = Util.count(clauses, clause -> superposition.subsumption.subsumed(clause));
            writer.printf(", %d live, %d subsumed (shown faded)", clauses.size() - subsumed, subsumed);
        }
        writer.println("</p>");
        writer.println("<code>");
        var limit = 50;
        var i = 0;
        for (; (i < clauses.size()) && (i < limit); i++) {
            var clause = clauses.get(i);
            printSample(clause);
        }
        var skip = clauses.size() - limit * 2;
        if (skip > 0) {
            for (var j = Integer.toString(skip).length(); j-- > 0; ) {
                writer.println(".<br>");
            }
            i += skip;
        }
        for (; i < clauses.size(); i++) {
            var clause = clauses.get(i);
            printSample(clause);
        }
        writer.println("</code>");
    }

    public static void printSummary(String file) throws IOException {
        open(Objects.toString(file, "summary"));
        var numberFormat = NumberFormat.getInstance();

        // Problems
        writer.println("<table class=\"bordered\">");
        writer.println("<tr>");
        writer.println("<th class=\"bordered\">Problem");
        writer.println("<th class=\"bordered\">Predicates");
        writer.println("<th class=\"bordered\">Functions");
        writer.println("<th class=\"bordered\">Formulas");
        writer.println("<th class=\"bordered\">Clauses");
        writer.println("<th class=\"bordered\">Generated");
        writer.println("<th class=\"bordered\">Discarded");
        writer.println("<th class=\"bordered\">Final");
        writer.println("<th class=\"bordered\">Expected");
        writer.println("<th class=\"bordered\">Result");
        writer.println("<th class=\"bordered\">Solved");
        writer.println("<th class=\"bordered\">Time");
        for (var problem : summaries) {
            writer.println("<tr>");
            writer.printf("<td class=\"bordered\"><a href=\"%s.html\">%s</a>\n", problem.name, problem.name);
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.predicates));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.functions));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.formulas));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.clauses));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.generated));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.discarded));
            writer.println("<td class=\"bordered\"; style=\"text-align: right\">" + numberFormat.format(problem.final1));
            writer.println("<td class=\"bordered\">" + Objects.toString(problem.expected, ""));
            writer.println("<td class=\"bordered\">" + Objects.toString(problem.result, ""));
            writer.println("<td class=\"bordered\"; style=\"text-align: center\">" + (problem.result.solved()
                                                                                      ? "&#x2714;"
                                                                                      : ""));
            writer.printf("<td class=\"bordered\" style=\"text-align: right;\">%.3f\n", problem.time / 1000.0);
        }
        writer.println("</table>");

        // Solved
        var solved = Util.count(summaries, problem -> problem.result.solved());
        writer.printf("<p>Solved %d/%d (%d%%)\n", solved, summaries.size(), solved * 100 / summaries.size());

        // Time
        writer.println("<p>" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy, HH:mm:ss")));

        // Version
        if (Main.version() != null) {
            writer.println("<p>Prover " + Main.version());
        }

        // End
        writer.close();
    }

    private static void quant(Term term) {
        writer.print('[');
        var params = term.get(1);
        for (var i = 0; i < params.size(); i++) {
            var x = params.get(i);
            if (i > 0) {
                writer.print(", ");
            }
            writer.print(x);
            if (x.type() != Type.INDIVIDUAL) {
                writer.print(':');
                print(x.type());
            }
        }
        writer.print("]: ");
        print(term.get(2), term);
    }

    public static Problem read(Path path) throws IOException {
        Function.nextId = 0;
        Formula.nextId = 0;
        var stream = System.in;
        if (!Util.stdin.equals(path)) {
            stream = new FileInputStream(path.toFile());
        }
        switch (language(path)) {
        case DIMACS:
            return DimacsParser.read(path, stream);
        case TPTP:
            return TptpParser.read(path, stream);
        }
        throw new IllegalStateException();
    }

    public void solve() {
        nextIdFunc = Function.nextId;
        nextIdFormula = Formula.nextId;
        timeReadEnd = System.currentTimeMillis();
        if (!formulas.isEmpty()) {
            cnf = new CNF(formulas, clauses);
        }
        try {
            superposition = new Superposition(clauses);
            superposition.start();
            superposition.join(Main.timeout);
            if (superposition.isAlive()) {
                superposition.interrupt();
                superposition.join();
            }
            result = superposition.result;
            if (conjecture != null) {
                switch (result) {
                case Satisfiable:
                    result = SZS.CounterSatisfiable;
                    break;
                case Unsatisfiable:
                    result = SZS.Theorem;
                    break;
                }
            }
            timeEnd = System.currentTimeMillis();
        } catch (InterruptedException e) {

            // Pro forma exception handler
            // that should never actually be triggered
            throw new IllegalStateException();
        }
    }

    public static SZS solve(List<Clause> clauses) {
        var problem = new Problem(clauses);
        problem.solve();
        return problem.result;
    }

    private String source(Formula formula) {
        if (formula.szs() == SZS.LogicalData) {
            if (formula.file() == null) {
                return "introduced(definition)";
            }
            return String.format("file(%s, %s)", Util.quote('\'', formula.file()), formula);
        }
        return String.format("inference(%s, [status(%s)], %s)",
                             formula.inference(),
                             formula.szs().abbreviation().toLowerCase(Locale.ROOT),
                             Arrays.toString(formula.from()));
    }

    private void wrap(String s) {
        var col = 0;
        for (var i = 0; i < s.length(); ) {
            var j = i;
            while ((j < s.length()) && (s.charAt(j) == ' ')) {
                j++;
            }
            var word = j;
            while ((j < s.length()) && (s.charAt(j) != ' ')) {
                j++;
            }
            j = Math.min(j, word + 90);
            if ((col > 0) && (col + (j - i) > 80)) {
                while ((i < j) && (s.charAt(i) == ' ')) {
                    i++;
                }
                writer.println();
                col = 0;
            }
            writer.print(s.substring(i, j).replace("<", "&lt;"));
            col += j - i;
            i = j;
        }
        writer.println();
    }

    private static final class InputFile {
        final Path path;
        final List<InputFile> includes = new ArrayList<>();

        InputFile(Path path) {
            this.path = path;
        }

        void print() {
            writer.printf("<a href=\"%s\">%s</a>", path.toString().replace('\\', '/'), path);
            if (includes.isEmpty()) {
                return;
            }
            writer.println("<ul>");
            for (var file : includes) {
                writer.println("<li>");
                file.print();
            }
            writer.println("</ul>");
        }
    }

    private static final class Summary {
        final String name;
        final int predicates;
        final int functions;
        final int formulas;
        final int clauses;
        final long generated;
        final long discarded;
        final int final1;
        final SZS expected;
        final SZS result;
        final long time;

        Summary(String name, int predicates, int functions, int formulas, int clauses, long generated, long discarded,
                int final1, SZS expected, SZS result, long time) {
            this.name = name;
            this.predicates = predicates;
            this.functions = functions;
            this.formulas = formulas;
            this.clauses = clauses;
            this.generated = generated;
            this.discarded = discarded;
            this.final1 = final1;
            this.expected = expected;
            this.result = result;
            this.time = time;
        }
    }
}
