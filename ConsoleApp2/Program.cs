using System;

class Program
{
    static float mul(float[] a, float[] b, int n)
    {
        float r = 0;
        for (int i = 0; i < n; i++)
            r += a[i] * b[i];
        return r;
    }

    static void Main(string[] args)
    {
        int n = 16;
        int count = 1_000_000_000;
        var a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = 1.0f;
        var b = new float[n];
        for (int i = 0; i < n; i++)
            b[i] = 1.0f;
        float x = 0.0f;
        while (count-- > 0)
            x += mul(a, b, n);
        Console.WriteLine(x);
    }
}
