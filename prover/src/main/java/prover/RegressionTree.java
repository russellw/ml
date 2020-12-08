package prover;

import java.io.IOException;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class RegressionTree {
    private RegressionTree() {}

    public static Term of(List<Record> records) {
        assert records.size() > 0;
        if (records.size() < 10) {
            return new Number(mean(records));
        }
        Variable minVariable = null;
        var minError = Double.POSITIVE_INFINITY;
        for (var variable : records.get(0).inputs.keySet()) {
            var error = error(records, variable);
            if (error < minError) {
                minVariable = variable;
                minError = error;
            }
        }
        assert minVariable != null;
        if (minVariable.type() == Type.NUMBER) {
            var threshold = threshold(records, minVariable);
            return split(records, minVariable, threshold);
        }
        return split(records, minVariable);
    }

    public static void test(String file) throws IOException {
        var data = new DataSet(file);
        data.print();

        // train
        var start = System.currentTimeMillis();
        var tree = of(data.training);
        System.out.println(tree);
        System.out.println((System.currentTimeMillis() - start) / 1000.0 + " seconds");

        // test
        System.out.println("naive error: " + error(data.testing));
        var error = 0.0;
        for (var record : data.testing) {
            var diff = record.output.number() - tree.eval(record.inputs).number();
            error += diff * diff;
        }
        System.out.println("dtree error: " + error);
    }

    private static double error(List<Record> records) {
        var mean = mean(records);
        var error = 0.0;
        for (var record : records) {
            var diff = record.output.number() - mean;
            error += diff * diff;
        }
        return error;
    }

    private static double error(List<Record> records, Variable variable) {
        if (variable.type() == Type.NUMBER) {
            var threshold = threshold(records, variable);
            return error(records, variable, threshold);
        }
        var error = 0.0;
        for (var value : variable.type().categories) {
            var filtered = filter(records, variable, value);
            error += error(filtered);
        }
        return error;
    }

    private static double error(List<Record> records, Variable variable, double threshold) {
        var lo = new ArrayList<Record>();
        var hi = new ArrayList<Record>();
        separate(records, variable, threshold, lo, hi);
        return error(lo) + error(hi);
    }

    private static List<Record> filter(List<Record> records, Variable variable, Category value) {
        var r = new ArrayList<Record>();
        for (var record : records) {
            if (record.inputs.get(variable) == value) {
                r.add(record);
            }
        }
        return r;
    }

    private static double mean(List<Record> records) {
        var n = 0;
        var total = 0.0;
        for (var record : records) {
            n++;
            total += record.output.number();
        }
        return total / n;
    }

    private static void separate(List<Record> records, Variable variable, double threshold, ArrayList<Record> lo,
                                 ArrayList<Record> hi) {
        for (var record : records) {
            ((record.inputs.get(variable).number() < threshold)
             ? lo
             : hi).add(record);
        }
    }

    private static Term split(List<Record> records, Variable variable) {
        var r = new ArrayList<Term>();
        r.add(variable);
        for (var value : variable.type().categories) {
            var filtered = filter(records, variable, value);
            if (filtered.isEmpty()) {
                r.add(new Number(mean(records)));
                continue;
            }
            r.add(of(filtered));
        }
        return new Case(r);
    }

    private static Term split(List<Record> records, Variable variable, double threshold) {
        var lo = new ArrayList<Record>();
        var hi = new ArrayList<Record>();
        separate(records, variable, threshold, lo, hi);
        Term ifTrue = lo.isEmpty()
                      ? new Number(mean(records))
                      : of(lo);
        Term ifFalse = lo.isEmpty()
                       ? new Number(mean(records))
                       : of(hi);
        return new IfLess(variable, threshold, ifTrue, ifFalse);
    }

    private static double threshold(List<Record> records, Variable variable) {
        records.sort(Comparator.comparingDouble(o -> o.inputs.get(variable).number()));
        var minThreshold = 0.0;
        var minError = Double.POSITIVE_INFINITY;
        for (int i = 0; i < records.size() - 1; i++) {
            var threshold = (records.get(i).inputs.get(variable).number() + records.get(i + 1).inputs.get(variable).number()) / 2;
            var error = error(records, variable, threshold);
            if (error < minError) {
                minThreshold = threshold;
                minError = error;
            }
        }
        return minThreshold;
    }
}
