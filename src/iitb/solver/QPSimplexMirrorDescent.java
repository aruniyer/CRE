package iitb.solver;

import iitb.shared.optimization.GradientCompute;
import iitb.shared.optimization.MirrorDescentTrainer;

import java.util.Arrays;

public class QPSimplexMirrorDescent extends AbstractQPSimplex {

    public static int MAX_ITERATIONS = 10000;
    public static double EPS_CONVERGENCE = 1e-12;

    public QPSimplexMirrorDescent() {
        super();
    }

    public QPSimplexMirrorDescent(double[][] A, double[] b) {
        super(A, b);
    }

    public QPSimplexMirrorDescent(double[][] A, double[] b, double[] x0) {
        super(A, b, x0);
    }

    public QPSimplexMirrorDescent(double[][] A, double[] b, int maxIteration, double epsConvergence) {
        super(A, b);
        MAX_ITERATIONS = maxIteration;
        EPS_CONVERGENCE = epsConvergence;
    }

    public QPSimplexMirrorDescent(double[][] A, double[] b, double[] x0, int maxIteration, double epsConvergence) {
        super(A, b, x0);
        MAX_ITERATIONS = maxIteration;
        EPS_CONVERGENCE = epsConvergence;
    }

    @Override
    public void solve() {
        GradientCompute gradientCompute = new QPSimplexGradientCompute(A, b);
        double[] x = new double[A.length];
        if (x0 == null) {
            Arrays.fill(x, 1.0f / x.length);
        } else {
            System.arraycopy(x0, 0, x, 0, x.length);
        }
        try {
            MirrorDescentTrainer mirrorDescentTrainer = new MirrorDescentTrainer(0, MAX_ITERATIONS, 1e-12);
            this.objective = mirrorDescentTrainer.optimize(x, gradientCompute);
            this.solution = x;
        } catch (Exception exception) {
            throw new RuntimeException("Exception with trainer!", exception);
        }
    }

    private class QPSimplexGradientCompute implements GradientCompute {

        double[][] A;
        double[] b;

        public QPSimplexGradientCompute(double[][] A, double[] b) {
            this.A = A;
            this.b = b;
        }

        @Override
        public double computeFunctionGradient(double[] lambda, double[] grad) throws Exception {
            double objective = 0;
            for (int i = 0; i < grad.length; i++)
                grad[i] = 0;
            for (int i = 0; i < grad.length; i++) {
                for (int j = 0; j < A[i].length; j++) {
                    grad[i] += A[i][j] * lambda[j];
                }
                objective += grad[i] * lambda[i] * (1.0 / 2.0) + b[i] * lambda[i];
                grad[i] = grad[i] + b[i];
            }
            return objective;
        }

    }

    public static void main(String[] args) {
        double[][] A = { { 2, 0.5 }, { 0.5, 1 } };
        double[][] A2 = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A.length; j++) {
                A2[i][j] = A[i][j] * 2;
            }
        }
        double[] b = { 1, 1 };
        AbstractQPSimplex solver = new QPSimplexMirrorDescent(A, b);
        solver.solve();
        System.out.println(Arrays.toString(solver.getSolution()));
        System.out.println(solver.getObjective());
    }

}
