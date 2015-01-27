package iitb.util;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.lang.mutable.MutableInt;

/**
 * 
 * Implementation of Chase's Twiddle algorithm to iterate through (N choose M) objects. 
 * https://www.deepdyve.com/lp/association-for-computing-machinery/algorithm-382-combinations-of-m-out-of-n-objects-g6-JkdUpWf2oW
 * 
 * @author aruniyer
 *
 */

public class Twiddle implements Iterable<int[]> {

    private int M, N;
    private int[] p;

    public Twiddle(int N, int M) {
        this.M = M;
        this.N = N;
        this.p = new int[N + 2];
    }

    @Override
    public Iterator<int[]> iterator() {
        return new TwiddleIterator(N, M, p);
    }

    private class TwiddleIterator implements Iterator<int[]> {

        private MutableInt x = new MutableInt(0), y = new MutableInt(0),
                z = new MutableInt(0);
        private int[] p;
        private int M, N;
        private int[] baseIndicator;

        TwiddleIterator(int n, int m, int[] p) {
            this.M = m;
            this.N = n;
            this.p = p;
            int i;
            p[0] = n + 1;
            for (i = 1; i != n - m + 1; i++) {
                p[i] = 0;
            }
            while (i != n + 1) {
                p[i] = i + m - n;
                i++;
            }
            p[n + 1] = -2;
            if (m == 0)
                p[1] = 1;
        }

        private boolean check(MutableInt x, MutableInt y, MutableInt z, int[] p) {
            int i, j, k;
            j = 1;
            while (p[j] <= 0)
                j++;
            if (p[j - 1] == 0) {
                for (i = j - 1; i != 1; i--)
                    p[i] = -1;
                p[j] = 0;
                x.setValue(0);
                z.setValue(0);
                p[1] = 1;
                y.setValue(j - 1);
            } else {
                if (j > 1)
                    p[j - 1] = 0;
                do
                    j++;
                while (p[j] > 0);
                k = j - 1;
                i = j;
                while (p[i] == 0)
                    p[i++] = -1;
                if (p[i] == -1) {
                    p[i] = p[k];
                    z.setValue(p[k] - 1);
                    x.setValue(i - 1);
                    y.setValue(k - 1);
                    p[k] = -1;
                } else {
                    if (i == p[0])
                        return true;
                    else {
                        p[j] = p[i];
                        z.setValue(p[i] - 1);
                        p[i] = 0;
                        x.setValue(j - 1);
                        y.setValue(i - 1);
                    }
                }
            }
            return false;
        }

        @Override
        public boolean hasNext() {
            return (baseIndicator == null || !check(x, y, z, p));
        }

        @Override
        public int[] next() {
            if (baseIndicator == null) {
                baseIndicator = new int[N];
                int i;
                for (i = 0; i != N - M; i++) {
                    baseIndicator[i] = 0;
                }
                while (i != N) {
                    baseIndicator[i++] = 1;
                }
            } else {
                baseIndicator[x.intValue()] = 1;
                baseIndicator[y.intValue()] = 0;
            }
            return baseIndicator;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

    }
    
    public static void main(String[] args) {
        Twiddle twiddle = new Twiddle(5, 3);
        for (int[] indicator : twiddle) {
            System.out.println(Arrays.toString(indicator));
        }
    }

}
