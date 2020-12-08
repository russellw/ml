package prover;

import java.io.*;

import java.nio.charset.StandardCharsets;

import java.util.*;
import java.util.function.BiFunction;

public final class DataSet {
    public final String file;

    // variables
    public final Variable[] variables;
    public final Variable[] inputVariables;
    public final Variable outputVariable;

    // records
    public final List<Record> records = new ArrayList<>();
    public final List<Record> training = new ArrayList<>();
    public final List<Record> testing = new ArrayList<>();

    public DataSet(String file) throws IOException {
        this.file = file;

        // data
        var data = readCsv(file);

        // variables
        variables = new Variable[data.get(0).length];
        for (int j = 0; j < variables.length; j++) {
            variables[j] = variable(data, j);
        }
        inputVariables = Arrays.copyOf(variables, variables.length - 1);
        outputVariable = variables[inputVariables.length];

        // records
        for (int i = 1; i < data.size(); i++) {
            records.add(record(data, i));
        }
        Collections.shuffle(records);
        var n = records.size() * 4 / 5;
        for (int i = 0; i < n; i++) {
            training.add(records.get(i));
        }
        for (int i = n; i < records.size(); i++) {
            testing.add(records.get(i));
        }
    }

    public void print() {
        System.out.println(file);
        var len = 0;
        for (var x : variables) {
            len = Math.max(len, x.toString().length());
        }
        for (var x : inputVariables) {
            print(len, x, (record, variable) -> record.inputs.get(variable));
        }
        print(len, outputVariable, (record, variable) -> record.output);
        System.out.printf("%d records = %d + %d\n", records.size(), training.size(), testing.size());
    }

    private static boolean isNumber(String s) {
        switch (s.charAt(0)) {
        case '-':
        case '.':
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
            return true;
        }
        return false;
    }

    private void print(int len, Variable x, BiFunction<Record, Variable, Term> f) {
        System.out.print(x);
        var width = len + 2 - x.toString().length();
        for (int i = 0; i < width; i++) {
            System.out.print(' ');
        }
        var type = x.type();
        if (type == Type.NUMBER) {
            var lo = 0.0;
            var hi = 0.0;
            for (var record : records) {
                var value = f.apply(record, x).number();
                lo = Math.min(lo, value);
                hi = Math.max(hi, value);
            }
            System.out.printf("%f - %f", lo, hi);
        } else {
            for (int i = 0; i < type.categories.length; i++) {
                if (i == 10) {
                    System.out.print("...");
                    break;
                }
                if (i > 0) {
                    System.out.print(", ");
                }
                System.out.print(type.categories[i]);
            }
        }
        System.out.println();
    }

    private static List<String[]> readCsv(String file) throws IOException {
        var reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
        var data = new ArrayList<String[]>();
        for (;;) {
            var line = reader.readLine();
            if (line == null) {
                break;
            }
            var r = line.split(",");
            for (int i = 0; i < r.length; i++) {
                r[i] = r[i].trim();
            }
            data.add(r);
        }
        var columns = data.get(0).length;
        for (var r : data) {
            if (r.length != columns) {
                throw new ParseException(file, "non-rectangular data");
            }
        }
        return data;
    }

    private Record record(List<String[]> data, int i) {
        var inputs = new HashMap<Variable, Term>();
        for (int j = 0; j < inputVariables.length; j++) {
            inputs.put(inputVariables[j], value(data, i, j));
        }
        return new Record(inputs, value(data, i, inputVariables.length));
    }

    private Term value(List<String[]> data, int i, int j) {
        var s = data.get(i)[j];
        var type = variables[j].type();
        if (type == Type.NUMBER) {
            return new Number(Double.parseDouble(s));
        }
        return type.category(s);
    }

    private static Variable variable(List<String[]> data, int j) {
        var header = data.get(0);
        var name = header[j];
        var values = new HashSet<String>();
        Type type = null;
        for (int i = 1; i < data.size(); i++) {
            var s = data.get(i)[j];
            if (isNumber(s)) {
                type = Type.NUMBER;
                break;
            }
            values.add(s);
        }
        if (type == null) {
            type = new Type(name, values);
        }
        return new Variable(type, name);
    }
}
