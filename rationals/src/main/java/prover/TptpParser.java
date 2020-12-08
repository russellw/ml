package prover;

import java.io.*;

import java.math.BigInteger;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import java.util.*;
import java.util.regex.Pattern;

public final class TptpParser {
    private static final Pattern STATUS_PATTERN = Pattern.compile("\\s*Status\\s*:\\s*(\\w+)");

    // Tokens
    private static final int DEFINED_WORD = -2;
    private static final int DISTINCT_OBJECT = -3;
    private static final int EQV = -4;
    private static final int IMPLIES = -5;
    private static final int IMPLIESR = -6;
    private static final int INTEGER = -7;
    private static final int NAND = -8;
    private static final int NOT_EQ = -9;
    private static final int NOR = -10;
    private static final int RATIONAL = -11;
    private static final int REAL = -12;
    private static final int VAR = -13;
    private static final int WORD = -14;
    private static final int XOR = -15;

    // Problem state
    private static Problem problem;
    private static Map<String, Type> types;
    private static Map<String, Function> functions;
    private static Map<String, DistinctObject> distinctObjects;

    // File state
    private final String file;
    private final LineNumberReader reader;
    private final Set<String> select;
    private int c;
    private boolean header;
    private int tok;
    private String tokString;
    private Map<String, Variable> bound = new HashMap<>();
    private Map<String, Variable> free = new HashMap<>();

    @SuppressWarnings("fallthrough")
    private TptpParser(Path path, InputStream stream, Set<String> select) throws IOException {
        problem.enter(path);
        file = path.getFileName().toString();
        reader = new LineNumberReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
        reader.setLineNumber(1);
        this.select = select;
        c = reader.read();
        header = true;
        lex();
        header = false;
        if ((problem.header.size() > 0) && !problem.header.get(problem.header.size() - 1).isEmpty()) {
            problem.header.add("");
        }
        while (tok != -1) {
            var s = word();
            expect('(');
            var name = name();
            Formula.reserveId(name);
            switch (s) {
            case "cnf": {
                expect(',');

                // Role
                var role = word();
                switch (role) {
                case "assumption":
                case "plain":
                case "unknown":
                    warn(role + ": interpreting as axiom");
                    break;
                case "axiom":
                case "corollary":
                case "definition":
                case "hypothesis":
                case "lemma":
                case "negated_conjecture":
                case "theorem":
                    break;
                default:
                    throw new ParseException(file, reader.getLineNumber(), role + ": invalid role for cnf");
                }
                expect(',');

                // Formula
                free.clear();
                var negative = new ArrayList<Term>();
                var positive = new ArrayList<Term>();
                var parens = eat('(');
                do {
                    var not = eat('~');
                    var a = infixUnary();
                    try {
                        a.typeInfer(Type.BOOLEAN);
                    } catch (TypeException e) {
                        e.printStackTrace();
                        throw new ParseException(file, reader.getLineNumber(), e.toString());
                    }
                    if (a.op() == Op.NOT) {
                        not = !not;
                        a = a.get(1);
                    }
                    (not
                     ? negative
                     : positive).add(a);
                } while (eat('|'));
                if (parens) {
                    expect(')');
                }
                if ((select != null) && !select.contains(name)) {
                    break;
                }
                problem.clauses.add(new ClauseInput(negative, positive, name, file));
                break;
            }
            case "fof":
            case "tff": {
                expect(',');

                // Role
                var role = word();
                expect(',');

                // Formula
                switch (role) {
                case "assumption":
                case "plain":
                case "unknown":
                    warn(role + ": interpreting as axiom");
                case "axiom":
                case "corollary":
                case "definition":
                case "hypothesis":
                case "lemma":
                case "negated_conjecture":
                case "theorem": {
                    Term a = formula();
                    if ((select != null) && !select.contains(name)) {
                        break;
                    }
                    problem.formulas.add(new FormulaTermInput(a, name, file));
                    break;
                }
                case "conjecture": {
                    Term a = formula();
                    if ((select != null) && !select.contains(name)) {
                        break;
                    }
                    if (problem.conjecture != null) {
                        throw new ParseException(file, reader.getLineNumber(), "multiple conjectures not supported");
                    }
                    problem.conjecture = new FormulaTermInputConjecture(a, name, file);
                    break;
                }
                case "type": {
                    var parens = 0;
                    while (eat('(')) {
                        parens++;
                    }
                    var funcName = word();
                    Function.reserveId(funcName);
                    expect(':');
                    if ((tok == DEFINED_WORD) && "$tType".equals(tokString)) {
                        lex();
                        if (tok == '>') {
                            throw new InappropriateException();
                        }
                        typeAtom(funcName);
                    } else {
                        var type = type();
                        var a = functions.get(funcName);
                        if (a == null) {
                            a = new Function(type, funcName);
                            functions.put(funcName, a);
                        } else if (!a.type().equals(type)) {
                            throw new ParseException(file, reader.getLineNumber(), "type mismatch");
                        }
                    }
                    while (parens-- > 0) {
                        expect(')');
                    }
                    break;
                }
                default:
                    throw new ParseException(file, reader.getLineNumber(), role + ": unknown role");
                }
                break;
            }
            case "include": {

                // Absolute path
                var path1 = Path.of(name);
                if (path1.isAbsolute()) {
                    include(path1);
                }

                // Include path specified by environment variable
                var tptp = System.getenv("TPTP");
                if (tptp == null) {
                    throw new ParseException(file, reader.getLineNumber(), "TPTP environment variable not set");
                }
                include(Path.of(tptp, name));
                break;
            }
            case "thf":
                throw new InappropriateException();
            default:
                throw new ParseException(file, reader.getLineNumber(), "unknown language");
            }
            if (tok == ',') {
                do {
                    ignore();
                } while (tok != ')');
            }
            expect(')');
            expect('.');
        }
        problem.leave();
    }

    private void args(List<Term> r) throws IOException {
        expect('(');
        do {
            r.add(atomicTerm(Type.INDIVIDUAL));
        } while (eat(','));
        expect(')');
    }

    private void args(List<Term> r, int arity) throws IOException {
        int n = r.size();
        args(r);
        n = r.size() - n;
        if (n != arity) {
            throw new ParseException(file, reader.getLineNumber(), "arg count: " + n + " != " + arity);
        }
    }

    private Term atomicTerm(Type defaultType) throws IOException {
        var k = tok;
        var s = tokString;
        lex();
        switch (k) {
        case '!':
        case '?':
        case '[':
            throw new InappropriateException();
        case DEFINED_WORD:
            switch (s) {
            case "$ceiling":
                return definedAtomicTerm(Term.CEIL, 1);
            case "$difference":
                return definedAtomicTerm(Term.SUBTRACT, 2);
            case "$distinct": {
                var r = new ArrayList<Term>();
                args(r);
                var inequalities = new ArrayList<Term>();
                inequalities.add(Term.AND);
                for (var i = 0; i < r.size(); i++) {
                    for (var j = 0; j < r.size(); j++) {
                        if (i != j) {
                            inequalities.add(r.get(i).notEq(r.get(j)));
                        }
                    }
                }
                return Term.of(inequalities);
            }
            case "$false":
                return Term.FALSE;
            case "$floor":
                return definedAtomicTerm(Term.FLOOR, 1);
            case "$greater": {
                var r = new ArrayList<Term>();
                args(r, 2);
                return r.get(1).less(r.get(0));
            }
            case "$greatereq": {
                var r = new ArrayList<Term>();
                args(r, 2);
                return r.get(1).lessEq(r.get(0));
            }
            case "$is_int":
                return definedAtomicTerm(Term.IS_INTEGER, 1);
            case "$is_rat":
                return definedAtomicTerm(Term.IS_RATIONAL, 1);
            case "$less":
                return definedAtomicTerm(Term.LESS, 2);
            case "$lesseq":
                return definedAtomicTerm(Term.LESS_EQ, 2);
            case "$product":
                return definedAtomicTerm(Term.MULTIPLY, 2);
            case "$quotient":
                return definedAtomicTerm(Term.DIVIDE, 2);
            case "$quotient_e":
                return definedAtomicTerm(Term.DIVIDE_EUCLIDEAN, 2);
            case "$quotient_f":
                return definedAtomicTerm(Term.DIVIDE_FLOOR, 2);
            case "$quotient_t":
                return definedAtomicTerm(Term.DIVIDE_TRUNCATE, 2);
            case "$remainder_e":
                return definedAtomicTerm(Term.REMAINDER_EUCLIDEAN, 2);
            case "$remainder_f":
                return definedAtomicTerm(Term.REMAINDER_FLOOR, 2);
            case "$remainder_t":
                return definedAtomicTerm(Term.REMAINDER_TRUNCATE, 2);
            case "$round":
                return definedAtomicTerm(Term.ROUND, 1);
            case "$sum":
                return definedAtomicTerm(Term.ADD, 2);
            case "$to_int":
                return definedAtomicTerm(Term.TO_INTEGER, 1);
            case "$to_rat":
                return definedAtomicTerm(Term.TO_RATIONAL, 1);
            case "$to_real":
                return definedAtomicTerm(Term.TO_REAL, 1);
            case "$true":
                return Term.TRUE;
            case "$truncate":
                return definedAtomicTerm(Term.TRUNCATE, 1);
            case "$uminus":
                return definedAtomicTerm(Term.NEGATE, 1);
            default:
                throw new ParseException(file, reader.getLineNumber(), s + ": unknown word");
            }
        case DISTINCT_OBJECT: {
            var a = distinctObjects.get(s);
            if (a == null) {
                a = new DistinctObject(s);
                distinctObjects.put(s, a);
            }
            return a;
        }
        case INTEGER:
            return Term.of(new BigInteger(s));
        case RATIONAL:
            return Term.of(BigRational.of(s));
        case REAL:

            // Real numbers are a problem
            // In general, they are incomputable
            // For computation purposes, double precision floating point is the best available approximation
            // However, theorem proving needs exactness
            // So represent real number literals not as the usual floating point
            // but as 'the real number that would correspond to this rational number'
            return Term.of(BigRational.ofDecimal(s)).toReal();
        case VAR: {
            var a = bound.get(s);
            if (a != null) {
                return a;
            }
            a = free.get(s);
            if (a != null) {
                return a;
            }
            a = new Variable(Type.INDIVIDUAL);
            free.put(s, a);
            return a;
        }
        case WORD: {
            Function.reserveId(s);
            if (tok == '(') {
                var r = new ArrayList<Term>();
                args(r);
                var a = functions.get(s);
                if (a == null) {
                    var returnType = (defaultType == null)
                                     ? new TypeVariable()
                                     : defaultType;
                    a = new Function(returnType, s, r);
                    functions.put(s, a);
                }
                return a.call(r);
            }
            var a = functions.get(s);
            if (a == null) {
                var type = (defaultType == null)
                           ? new TypeVariable()
                           : defaultType;
                a = new Function(type, s);
                functions.put(s, a);
            }
            return a;
        }
        default:
            throw new ParseException(file, reader.getLineNumber(), string(k, s) + ": term expected");
        }
    }

    private Term definedAtomicTerm(Term op, int arity) throws IOException {
        var r = new ArrayList<Term>();
        r.add(op);
        args(r, arity);
        return Term.of(r);
    }

    private boolean eat(int k) throws IOException {
        if (tok == k) {
            lex();
            return true;
        }
        return false;
    }

    private void expect(int k) throws IOException {
        if (eat(k)) {
            return;
        }
        throw new ParseException(file, reader.getLineNumber(), string(tok, tokString) + ": '" + (char) k + "' expected");
    }

    private Term formula() throws IOException {
        bound.clear();
        free.clear();
        var a = logicFormula();
        if (!free.isEmpty()) {
            throw new ParseException(file, reader.getLineNumber(), "unbound variable");
        }
        assert a.freeVars().isEmpty();
        try {
            a.typeInfer(Type.BOOLEAN);
        } catch (TypeException e) {
            e.printStackTrace();
            throw new ParseException(file, reader.getLineNumber(), e.toString());
        }
        return a;
    }

    private void ignore() throws IOException {
        switch (tok) {
        case '(':
            lex();
            while (!eat(')')) {
                ignore();
            }
            break;
        case '[':
            lex();
            while (!eat(']')) {
                ignore();
            }
            break;
        case -1:
            throw new ParseException(file, reader.getLineNumber(), "unexpected end of file");
        default:
            lex();
            break;
        }
    }

    private void include(Path path) throws IOException {
        var stream = new FileInputStream(path.toFile());
        var select1 = select;
        if (eat(',')) {
            if ((tok == WORD) && "all".equals(tokString)) {
                lex();
            } else {
                expect('[');
                select1 = new HashSet<>();
                do {
                    var name = name();
                    if ((select == null) || select.contains(name)) {
                        select1.add(name);
                    }
                } while (eat(','));
                expect(']');
            }
        }
        new TptpParser(path, stream, select1);
    }

    private Term infixUnary() throws IOException {
        var a = atomicTerm(null);
        switch (tok) {
        case '=':
            lex();
            return a.eq(atomicTerm(null));
        case NOT_EQ:
            lex();
            return a.notEq(atomicTerm(null));
        default:
            return a;
        }
    }

    private void lex() throws IOException {
        for (;;) {
            tok = c;
            switch (c) {
            case ' ':
            case '\f':
            case '\n':
            case '\r':
            case '\t':
                c = reader.read();
                continue;
            case '!':
                c = reader.read();
                if (c == '=') {
                    c = reader.read();
                    tok = NOT_EQ;
                    break;
                }
                break;
            case '"':
                lexQuote();
                tok = DISTINCT_OBJECT;
                break;
            case '$':
                var sb = new StringBuilder();
                do {
                    sb.append((char) c);
                    c = reader.read();
                } while (Character.isJavaIdentifierPart(c));
                tok = DEFINED_WORD;
                tokString = sb.toString();
                break;
            case '%':
                var s = reader.readLine();
                c = reader.read();
                if (header) {
                    problem.header.add('%' + s);
                    if (c == '\n') {
                        problem.header.add("");
                    }
                }
                if (problem.expected == null) {
                    var matcher = STATUS_PATTERN.matcher(s);
                    if (matcher.matches()) {
                        problem.expected = SZS.valueOf(matcher.group(1));
                    }
                }
                continue;
            case '+':
            case '-':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                sb = new StringBuilder();
                do {
                    sb.append((char) c);
                    c = reader.read();
                } while (Character.isDigit(c));
                switch (c) {
                case '.':
                    do {
                        sb.append((char) c);
                        c = reader.read();
                    } while (Character.isDigit(c));
                    break;
                case '/':
                    do {
                        sb.append((char) c);
                        c = reader.read();
                    } while (Character.isDigit(c));
                    tok = RATIONAL;
                    tokString = sb.toString();
                    return;
                case 'E':
                case 'e':
                    break;
                default:
                    tok = INTEGER;
                    tokString = sb.toString();
                    return;
                }
                if ((c == 'e') || (c == 'E')) {
                    sb.append((char) c);
                    c = reader.read();
                }
                if ((c == '+') || (c == '-')) {
                    sb.append((char) c);
                    c = reader.read();
                }
                while (Character.isDigit(c)) {
                    sb.append((char) c);
                    c = reader.read();
                }
                tok = REAL;
                tokString = sb.toString();
                break;
            case '/':
                var line = reader.getLineNumber();
                c = reader.read();
                if (c != '*') {
                    throw new ParseException(file, reader.getLineNumber(), "'*' expected");
                }
                do {
                    do {
                        if (c == -1) {
                            throw new ParseException(file, line, "unclosed block comment");
                        }
                        c = reader.read();
                    } while (c != '*');
                    c = reader.read();
                } while (c != '/');
                c = reader.read();
                continue;
            case '<':
                c = reader.read();
                if (c == '=') {
                    c = reader.read();
                    if (c == '>') {
                        c = reader.read();
                        tok = EQV;
                        break;
                    }
                    tok = IMPLIESR;
                    break;
                }
                if (c == '~') {
                    c = reader.read();
                    if (c == '>') {
                        c = reader.read();
                        tok = XOR;
                        break;
                    }
                    throw new ParseException(file, reader.getLineNumber(), "'>' expected");
                }
                break;
            case '=':
                c = reader.read();
                if (c == '>') {
                    c = reader.read();
                    tok = IMPLIES;
                    break;
                }
                break;
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'E':
            case 'F':
            case 'G':
            case 'H':
            case 'I':
            case 'J':
            case 'K':
            case 'L':
            case 'M':
            case 'N':
            case 'O':
            case 'P':
            case 'Q':
            case 'R':
            case 'S':
            case 'T':
            case 'U':
            case 'V':
            case 'W':
            case 'X':
            case 'Y':
            case 'Z':
                sb = new StringBuilder();
                do {
                    sb.append((char) c);
                    c = reader.read();
                } while (Character.isJavaIdentifierPart(c));
                tok = VAR;
                tokString = sb.toString();
                break;
            case '\'':
                lexQuote();
                if (tokString.length() == 0) {
                    throw new ParseException(file, reader.getLineNumber(), "empty word");
                }
                tok = WORD;
                break;
            case 'a':
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
            case 'g':
            case 'h':
            case 'i':
            case 'j':
            case 'k':
            case 'l':
            case 'm':
            case 'n':
            case 'o':
            case 'p':
            case 'q':
            case 'r':
            case 's':
            case 't':
            case 'u':
            case 'v':
            case 'w':
            case 'x':
            case 'y':
            case 'z':
                sb = new StringBuilder();
                do {
                    sb.append((char) c);
                    c = reader.read();
                } while (Character.isJavaIdentifierPart(c));
                tok = WORD;
                tokString = sb.toString();
                break;
            case '~':
                c = reader.read();
                if (c == '&') {
                    c = reader.read();
                    tok = NAND;
                    break;
                }
                if (c == '|') {
                    c = reader.read();
                    tok = NOR;
                    break;
                }
                break;
            default:
                c = reader.read();
                break;
            }
            return;
        }
    }

    private void lexQuote() throws IOException {
        var line = reader.getLineNumber();
        var quote = c;
        var sb = new StringBuilder();
        c = reader.read();
        while (c != quote) {
            if (c < ' ') {
                throw new ParseException(file, line, "unclosed quote");
            }
            if (c == '\\') {
                c = reader.read();
            }
            sb.append((char) c);
            c = reader.read();
        }
        c = reader.read();
        tokString = sb.toString();
    }

    private Term logicFormula() throws IOException {
        var a = unaryFormula();
        switch (tok) {
        case '&':
            return logicFormula1(Term.AND, a);
        case '|':
            return logicFormula1(Term.OR, a);
        case EQV:
            lex();
            return a.eqv(unaryFormula());
        case IMPLIES:
            lex();
            return a.implies(unaryFormula());
        case IMPLIESR:
            lex();
            return unaryFormula().implies(a);
        case NAND:
            lex();
            return a.nand(unaryFormula());
        case NOR:
            lex();
            return a.nor(unaryFormula());
        case XOR:
            lex();
            return a.xor(unaryFormula());
        default:
            return a;
        }
    }

    private Term logicFormula1(Term op, Term a) throws IOException {
        var k = tok;
        var r = new ArrayList<Term>();
        r.add(op);
        r.add(a);
        while (eat(k)) {
            r.add(unaryFormula());
        }
        return Term.of(r);
    }

    private String name() throws IOException {
        switch (tok) {
        case INTEGER:
        case WORD:
            break;
        default:
            throw new ParseException(file, reader.getLineNumber(), "name expected");
        }
        var s = tokString;
        lex();
        return s;
    }

    public static Problem read(Path path, InputStream stream) throws IOException {
        types = new HashMap<>();
        functions = new HashMap<>();
        distinctObjects = new HashMap<>();
        problem = new Problem();

        // Read
        new TptpParser(path, stream, null);

        // Negate conjecture
        if (problem.conjecture != null) {
            var negatedConjecture = new FormulaTermFrom(problem.conjecture.term().not(), problem.conjecture) {
                @Override
                public String inference() {
                    return "negate";
                }
            };
            negatedConjecture.setId();
            problem.formulas.add(negatedConjecture);
        }

        // Free memory
        types = null;
        functions = null;
        distinctObjects = null;

        // Return
        return problem;
    }

    private String string(int k, String s) {
        switch (k) {
        case DEFINED_WORD:
        case VAR:
            return s;
        case DISTINCT_OBJECT:
            return Util.quote('"', s);
        case EQV:
            return "<=>";
        case IMPLIES:
            return "=>";
        case IMPLIESR:
            return "<=";
        case NAND:
            return "~&";
        case NOR:
            return "~|";
        case NOT_EQ:
            return "!=";
        case WORD:
            return Util.quote('\'', s);
        case XOR:
            return "<~>";
        default:
            assert k > ' ';
            return '\'' + Character.toString(k) + '\'';
        }
    }

    private Type type() throws IOException {
        if (eat('(')) {
            var r = new ArrayList<Type>();
            r.add(null);
            do {
                r.add(typeAtom());
            } while (eat('*'));
            expect(')');
            expect('>');
            var returnType = typeAtom();
            r.set(0, returnType);
            return Type.of(r);
        }
        var type = typeAtom();
        if (eat('>')) {
            var returnType = typeAtom();
            return Type.of(returnType, type);
        }
        return type;
    }

    private Type typeAtom() throws IOException {
        var k = tok;
        var s = tokString;
        lex();
        switch (k) {
        case '!':
        case '[':
            throw new InappropriateException();
        case DEFINED_WORD:
            switch (s) {
            case "$i":
                return Type.INDIVIDUAL;
            case "$int":
                return Type.INTEGER;
            case "$o":
                return Type.BOOLEAN;
            case "$rat":
                return Type.RATIONAL;
            case "$real":
                return Type.REAL;
            case "$tType":
                throw new InappropriateException();
            default:
                throw new ParseException(file, reader.getLineNumber(), s + ": unknown type");
            }
        case WORD:
            return typeAtom(s);
        default:
            throw new ParseException(file, reader.getLineNumber(), "type expected");
        }
    }

    private static Type typeAtom(String name) {
        var type = types.get(name);
        if (type != null) {
            return type;
        }
        type = new TypeConstant(name);
        types.put(name, type);
        return type;
    }

    private Term unaryFormula() throws IOException {
        switch (tok) {
        case '!': {
            var old = new HashMap<>(bound);
            var a = Term.of(Term.ALL, unaryFormulaParams(), unaryFormula());
            bound = old;
            return a;
        }
        case '(': {
            lex();
            var a = logicFormula();
            expect(')');
            return a;
        }
        case '?': {
            var old = new HashMap<>(bound);
            var a = Term.of(Term.EXISTS, unaryFormulaParams(), unaryFormula());
            bound = old;
            return a;
        }
        case '~':
            lex();
            return unaryFormula().not();
        default:
            return infixUnary();
        }
    }

    private Term unaryFormulaParams() throws IOException {
        lex();
        expect('[');
        List<Variable> params = new ArrayList<>();
        do {
            if (tok != VAR) {
                throw new ParseException(file, reader.getLineNumber(), "variable expected");
            }
            var name = tokString;
            lex();
            var type = eat(':')
                       ? typeAtom()
                       : Type.INDIVIDUAL;
            Variable a = new Variable(type);
            bound.put(name, a);
            params.add(a);
        } while (eat(','));
        expect(']');
        expect(':');
        return Term.of(params);
    }

    private void warn(String message) {
        System.err.printf("%s:%d: warning: %s\n", file, reader.getLineNumber(), message);
    }

    private String word() throws IOException {
        if (tok != WORD) {
            throw new ParseException(file, reader.getLineNumber(), "word expected");
        }
        var s = tokString;
        lex();
        return s;
    }
}
