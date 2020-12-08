package prover;

import java.util.Random;
import java.util.function.DoubleSupplier;

public final class Optimizer {
    private final Param[] params;
    private final DoubleSupplier f;
    private Random random = new Random();

    private Optimizer(Param[] params, DoubleSupplier f) {
        this.params = params;
        this.f = f;
    }

    private int[] get() {
        var values = new int[params.length];
        for (var i = 0; i < params.length; i++) {
            var param = params[i];
            var value = param.get();
            values[i] = value;
        }
        return values;
    }

    public static void optimize(Param[] params, DoubleSupplier f) {
        new Optimizer(params, f).run();
    }

    private int[] random() {
        var values = new int[params.length];
        for (var i = 0; i < params.length; i++) {
            var param = params[i];
            var value = param.min + random.nextInt(param.max - param.min + 1);
            values[i] = value;
        }
        return values;
    }

    private void run() {
        var bestValues = get();
        var bestScore = test(0, bestValues);
        for (int i = 1; i <= 10; i++) {
            var values = random();
            var score = test(i, values);
            if (score > bestScore) {
                bestValues = values;
                bestScore = score;
            }
        }
        set(bestValues);
    }

    private void set(int[] values) {
        for (var i = 0; i < params.length; i++) {
            var param = params[i];
            var value = values[i];
            param.set(value);
        }
    }

    private double test(int i, int[] values) {
        System.out.printf("%3d", i);
        for (var value : values) {
            System.out.printf(" %7d", value);
        }
        set(values);
        return f.getAsDouble();
    }

    public static abstract class Param {
        final int min, max;

        public Param(int min, int max) {
            this.min = min;
            this.max = max;
        }

        abstract int get();

        abstract String name();

        abstract void set(int value);
    }
}
