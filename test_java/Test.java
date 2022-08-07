// results on a 3 GHz CPU
// counting only FP multiplies/sec

// vector length 10
// 1587e6

// vector length 100
// 1369e6

import java.lang.*;

class Test {
    static float mul(float[] a, float[] b, int n)
    {
        float r = 0;
        for (int i = 0; i < n; i++)
            r += a[i] * b[i];
        return r;
    }

    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        int count = Integer.parseInt(args[1]);
        var a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = 1.0f;
        var b = new float[n];
        for (int i = 0; i < n; i++)
            b[i] = 1.0f;
        float x = 0.0f;
        while (count-- > 0)
            x += mul(a, b, n);
        System.out.println(x);
    }
}
