package iitb.solver;

import java.util.Arrays;

/**
 * QP Simplex solves QP with a simplex constraint on the variable. QP Form is given as:
 * 
 * <pre>
 * (1/2)x^TAx + b^Tx 
 * subject to: 1^Tx = 1
 * </pre>
 * 
 * @author aruniyer
 * 
 */
public abstract class AbstractQPSimplex implements IConvexQP {

    protected double[][] A;
    protected double[] b;
    protected double[] x0;
    protected double[] solution;
    protected double objective;

    public AbstractQPSimplex() {
        this(null, null);
    }

    public AbstractQPSimplex(double[][] A, double[] b) {
        this(A, b, null);
        if (A != null) {
            this.x0 = new double[A.length];
            Arrays.fill(x0, 1 / x0.length);
        }
    }

    public AbstractQPSimplex(double[][] A, double[] b, double[] x0) {
        this.A = A;
        this.b = b;
        this.setInitialSolution(x0);
    }

    @Override
    public void setA(double[][] A) {
        this.A = A;
    }

    @Override
    public void setb(double[] b) {
        this.b = b;
    }

    @Override
    public void setInitialSolution(double[] x0) {
        this.x0 = x0;
    }

    @Override
    public double[] getSolution() {
        return solution;
    }

    @Override
    public double getObjective() {
        return objective;
    }

}