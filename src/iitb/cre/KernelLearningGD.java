package iitb.cre;

import java.util.Arrays;

import Jama.Matrix;
import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.kernel.IKernelFunction;
import iitb.shared.optimization.GradientCompute;
import iitb.solver.QPSimplexMirrorDescent;

/**
 * 
 * Kernel Learning from ICML 2014 solved via projected subgradient descent algorithm
 * 
 * @author aruniyer
 *
 */
public class KernelLearningGD extends AbstractKernelLearning implements GradientCompute {

    private QPSimplexMirrorDescent qpSimplexSolver;

    public KernelLearningGD(IDataStore D, IDataStore[] Us, IKernelFunction[] kernelFunctions, float C1, float C2) {
        super(D, Us, kernelFunctions, C1, C2);
        this.qpSimplexSolver = new QPSimplexMirrorDescent();
    }

    @Override
    public void learnWeights() {
        double[] x = new double[kernelFunctions.length];
        Arrays.fill(x, 1);
        System.out.println("Doing projected subgradient descent ... ");
        double[] minEstimate = x.clone();
        try {
            // subgradient descent
            double alpha_k = 0.1;
            double[] gk = new double[x.length];
            this.computeFunctionGradient(x, gk);
            double R = 0;
            for (int i = 0; i < gk.length; i++) {
                R += gk[i] * gk[i];
            }
            R = alpha_k * 1000 * Math.sqrt(R);
            double normgk = 0;
            double fkbest = Double.MAX_VALUE / 4;
            double lkbest = -Double.MAX_VALUE / 4;
            double lk = 0;
            double sumAlphak = 0;
            
            double prevGap = Double.MAX_VALUE;
            int j;
            double fk = -1;
            for (j = 1; j <= 1000 && prevGap - fkbest + lkbest > 1e-2; j++) {
                prevGap = fkbest - lkbest;
                Arrays.fill(gk, 0);
                fk = this.computeFunctionGradient(x, gk);
                if (fk < fkbest) {
                    fkbest = fk;
                    minEstimate = x.clone();
                }
                
                normgk = 0;
                for (int k = 0; k < gk.length; k++) {
                    normgk += gk[k] * gk[k];
                }
                normgk = Math.sqrt(normgk);
                
                lk = (lk * 2 * sumAlphak + 2 * alpha_k * fk - alpha_k * alpha_k * normgk * normgk) / (2 * (sumAlphak + alpha_k));
                if (j == 1)
                    lk -= R * R / (2 * alpha_k);
                sumAlphak += alpha_k;
                lkbest = Math.max(lkbest, lk);
                
                for (int i = 0; i < x.length; i++) {
                    x[i] = Math.max(0, x[i] - alpha_k * gk[i]);
                }

                if (j%100 == 0 || j == 1)
                    System.out.println("iter =  " + j + ", fk = " + fk + " :: gradnorm = " + normgk + " :: fkbest = " + fkbest + " :: lkbest = " + lkbest);
            }
            System.out.println("iter =  " + j + ", fk = " + fk + " :: gradnorm = " + normgk + " :: fkbest = " + fkbest + " :: lkbest = " + lkbest);
        } catch (Exception exception) {
            throw new RuntimeException("Error while learning weights!", exception);
        }        
        this.weights = new float[x.length];
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = (float) minEstimate[i];
        }
        System.out.println("optimization complete.");
    }

    @Override
    public double computeFunctionGradient(double[] lambda, double[] grad) throws Exception {
        double[] coeffs = new double[grad.length];
        double obj = computeFirstGradient(lambda, coeffs);
        System.arraycopy(coeffs, 0, grad, 0, grad.length);
        if (C1 != 0) {
            obj += C1 * computeSecondGradient(lambda, coeffs);
            for (int i = 0; i < grad.length; i++) {
                grad[i] += C1 * coeffs[i];
            }
        }
        if (C2 != 0) {
            for (int u = 0; u < Us.length; u++) {
                obj += C2 * computeThirdGradient(u, lambda, coeffs);
                for (int i = 0; i < grad.length; i++) {
                    grad[i] += C2 * coeffs[i];
                }
            }
        }
        return obj;
    }

    private double computeFirstGradient(double[] weights, double[] coeffs) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            coeffs[i] = Math.signum(weights[i]);
            sum += Math.abs(weights[i]);
        }
        return sum;
    }

    private double computeSecondGradient(double[] weights, double[] coeffs) {
        int numY = D.numClasses();
        double[][] wtA = new double[numY][numY];
        for (int j = 0; j < wtA.length; j++) {
            for (int k = 0; k < wtA.length; k++) {
                for (int kernel = 0; kernel < kernelFunctions.length; kernel++) {
                    wtA[j][k] += weights[kernel] * A[kernel][j][k];
                }
            }
        }
        Matrix matrix = new Matrix(wtA);
        matrix = matrix.times(-1);
        double[] eigenValues = matrix.eig().getRealEigenvalues();
        double maxEig = -Double.MAX_VALUE;
        int index = -1;
        for (int i = 0; i < eigenValues.length; i++) {
            if (eigenValues[i] > 0) {
                System.out.println("weights = " + Arrays.toString(weights));
                throw new IllegalStateException();
            }
            if (maxEig < eigenValues[i]) {
                maxEig = eigenValues[i];
                index = i;
            }
        }
        float[] eigenVector = new float[wtA.length];
        for (int i = 0; i < wtA.length; i++) {
            eigenVector[i] = (float) matrix.eig().getV().get(i, index);
        }

        for (int kernel = 0; kernel < kernelFunctions.length; kernel++) {
            coeffs[kernel] = -getQObj(A[kernel], new float[wtA.length], eigenVector);
        }
        return maxEig;
    }

    private double computeThirdGradient(int bucket, double[] weights, double[] coeffs) {
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

        float maxAbs = 0;
        for (int i = 0; i < theta_i.length; i++) {
            maxAbs = Math.max(maxAbs, (float) Math.abs(theta[i] - theta_i[i]));
        }

        float[] diffTheta = new float[theta.length];
        for (int j = 0; j < theta.length; j++) {
            diffTheta[j] = (float) (theta_i[j] - theta[j]);
        }
        double[] compCoeffs = new double[coeffs.length];
        double objective = maxAbs;
        for (int kernel = 0; kernel < A.length; kernel++) {
            compCoeffs[kernel] = -getQObj(A[kernel], b[bucket][kernel], diffTheta);
            objective += compCoeffs[kernel] * weights[kernel];
        }
        if (objective > 0) {
            System.arraycopy(compCoeffs, 0, coeffs, 0, coeffs.length);
        } else {
            for (int i = 0; i < coeffs.length; i++)
                coeffs[i] = 0;
        }
        return Math.max(0, objective);
    }

}
