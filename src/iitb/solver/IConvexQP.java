package iitb.solver;

/**
 * A generic interface for convex QP solvers
 * 
 * @author aruniyer
 *
 */
public interface IConvexQP {

    public void setA(double[][] A);

    public void setb(double[] b);

    public void solve();

    public void setInitialSolution(double[] x0);

    public double[] getSolution();

    public double getObjective();

}
