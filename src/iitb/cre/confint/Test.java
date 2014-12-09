package iitb.cre.confint;

import org.apache.commons.math3.analysis.solvers.LaguerreSolver;
import org.apache.commons.math3.complex.Complex;

public class Test {
    
    public static void main(String[] args) {
        LaguerreSolver solver = new LaguerreSolver();
        Complex[] solutions = solver.solveAllComplex(new double[]{-4, 0, 1}, 0);
        for (Complex solution : solutions) {
            System.out.println(solution);
        }
    }

}
