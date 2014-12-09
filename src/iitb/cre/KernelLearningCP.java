package iitb.cre;

import java.util.Arrays;
import java.util.Set;

import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.kernel.IKernelFunction;
import iitb.solver.AbstractCuttingPlane;
import iitb.solver.AbstractQPSimplex;
import iitb.solver.IConstraint;
import iitb.solver.LinearConstraint;
import iitb.solver.QPSimplexMirrorDescent;

/**
 * 
 * Kernel Learning from ICML 2014 solved via cutting plane algorithm
 * 
 * @author aruniyer
 *
 */
public class KernelLearningCP extends AbstractKernelLearning {

    private CPSolver cpSolver;

    public KernelLearningCP(IDataStore D, IDataStore[] Us, IKernelFunction[] kernelFunctions, float C1, float C2) {
        super(D, Us, kernelFunctions, C1, C2);
        this.cpSolver = new CPSolver();
    }

    public void learnWeights() {
        float[] x = new float[2 * kernelFunctions.length];
        Arrays.fill(x, 1);
//        this.cpSolver.solve(x);
        this.weights = new float[kernelFunctions.length];
        System.arraycopy(x, 0, weights, 0, weights.length);
        throw new UnsupportedOperationException("Cutting plane solver is incomplete, please use Gradient Descent solver for the moment!");
    }

    private class CPSolver extends AbstractCuttingPlane {

        private AbstractQPSimplex qpSimplexSolver;

        CPSolver() {
            super(Us.length);
            this.qpSimplexSolver = new QPSimplexMirrorDescent();
        }

        protected IConstraint constraintGenerator(int bucket, float[] solution) {
            float[] weights = new float[kernelFunctions.length];
            System.arraycopy(solution, 0, weights, 0, weights.length);
            float[] xi = new float[kernelFunctions.length];
            System.arraycopy(solution, 0, xi, weights.length, xi.length);
            return constraintGenerator(bucket, weights, xi);
        }

        private IConstraint constraintGenerator(int bucket, float[] weights, float[] xi) {
            IDataStore U = Us[bucket];
            int numY = D.numClasses();
            double[] theta_i = new double[numY];
            for (IInstance instance : U) {
                int label = (int) instance.label();
                theta_i[label]++;
            }
            for (int j = 0; j < theta_i.length; j++) {
                theta_i[j] = theta_i[j] / U.size();
            }

            double[][] wtA = new double[numY][numY];
            double[] wtb = new double[numY];
            for (int j = 0; j < wtA.length; j++) {
                for (int k = 0; k < wtA.length; k++) {
                    for (int kernel = 0; kernel < kernelFunctions.length; kernel++) {
                        wtA[j][k] += weights[kernel] * A[kernel][j][k];
                    }
                }
                for (int kernel = 0; kernel < kernelFunctions.length; kernel++) {
                    wtb[j] += weights[kernel] * b[bucket][kernel][j];
                }
            }

            float minObjective = Float.MAX_VALUE;
            double[] theta = null;
            for (int clazz = 0; clazz < numY; clazz++) {
                for (int possibility = -1; possibility < 2; possibility += 2) {
                    double[] tempB = new double[wtb.length];
                    System.arraycopy(wtb, 0, tempB, 0, wtb.length);
                    tempB[clazz] = tempB[clazz] - 2 * possibility;
                    this.qpSimplexSolver.setA(wtA);
                    this.qpSimplexSolver.setb(tempB);
                    this.qpSimplexSolver.solve();
                    float objective = (float) this.qpSimplexSolver.getObjective();
                    if (objective < minObjective) {
                        minObjective = objective;
                        theta = this.qpSimplexSolver.getSolution();
                    }
                }
            }

            float checkValue = 0;
            float maxAbs = 0;
            for (int j = 0; j < wtA.length; j++) {
                for (int k = 0; k < wtA.length; k++) {
                    checkValue += (theta[j] - theta_i[j]) * (theta[k] - theta_i[k]) * wtA[j][k];
                }
                maxAbs = (float) Math.max(maxAbs, Math.abs(theta[j] - theta_i[j]));
            }
            checkValue += xi[bucket] - maxAbs;
            LinearConstraint constraint = null;
            if (checkValue < -1e-4) {
                constraint = new LinearConstraint(2 * kernelFunctions.length);
                for (int kernel = 0; kernel < kernelFunctions.length; kernel++) {
                    float[] diffTheta = new float[theta.length];
                    for (int j = 0; j < theta.length; j++) {
                        diffTheta[j] = (float) (theta[j] - theta_i[j]);
                    }
                    constraint.coefficients[kernel] = getQObj(A[kernel], b[bucket][kernel], diffTheta);
                }
                constraint.coefficients[weights.length + bucket] = 1;
                constraint.constant = maxAbs;
            }
            return constraint;
        }

        @Override
        protected void solveWithConstraints(Set<IConstraint> constraintSet, float[] initial) {
            // For the moment implement the lp solver without maxmineig objective
        }

    }

}
