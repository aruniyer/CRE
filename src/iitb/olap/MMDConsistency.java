package iitb.olap;

import java.util.Arrays;
import java.util.LinkedHashMap;

import Jama.Matrix;
import gnu.trove.list.array.TIntArrayList;
import iitb.olap.Config.DataHandler;
import iitb.shared.optimization.GradientCompute;
import iitb.shared.optimization.MirrorDescentTrainer;

public class MMDConsistency {

    enum WeightScheme {
        ROOTSCHEME, 
        PROPWEIGHTSCHEME, INVPROPWEIGHTSCHEME, 
        PROPCHILDSCHEME, PROPCHILDWEIGHTSCHEME, PROPCHILDINVWEIGHTSCHEME, 
        DISPARITYSCHEME, 
        ALLEQUAL
    }

    private double[][] A;
    private double[][] b;
    private int[] nu;
    private DataHandler dataHandler;
    private double[] estimates;
    private double objective;
    private WeightScheme scheme;

    public MMDConsistency(double[][] A, double[][] b, int[] nu, WeightScheme scheme, DataHandler dataHandler) throws Exception {
        this.A = A;
        this.b = b;
        this.nu = nu;
        this.dataHandler = dataHandler;

        this.scheme = scheme;
        if (this.scheme.equals(WeightScheme.DISPARITYSCHEME)) {
            doWeightLearning();
        }
        double[] x = new double[b.length * A.length];
        Arrays.fill(x, 1.0 / A.length);
        this.objective = doBlockCoordinateGradientDescent(x);
        this.estimates = x;
    }
    
    private LinkedHashMap<TIntArrayList, Integer> groupIndex;
    private double[] weights;
    private void doWeightLearning() {
	System.out.println("Learning weights for disparity ... ");
	System.out.println("Building index for group ... ");
        groupIndex = new LinkedHashMap<TIntArrayList, Integer>();
        int indexer = 0;
        for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                groupIndex.put(new TIntArrayList(baseIndicator), indexer);
                indexer++;
            }
        }
        System.out.println("Total number of groups : " + indexer);
        weights = new double[indexer];
        Arrays.fill(weights, 1);
        double[] x = new double[weights.length];
        Arrays.fill(x, 1);
        GradientCompute gradientCompute = new DisparityGradientComputer(100);
        System.out.println("Doing projected subgradient descent ... ");
        double[] minEstimate = x.clone();
        try {
            System.out.println("start");
            // subgradient descent
            double alpha_k = 0.1;
            double[] gk = new double[x.length];
            gradientCompute.computeFunctionGradient(x, gk);
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
                System.out.println("1+");
                prevGap = fkbest - lkbest;
                Arrays.fill(gk, 0);
                fk = gradientCompute.computeFunctionGradient(x, gk);
                System.out.println("1+");
                if (fk < fkbest) {
                    fkbest = fk;
                    minEstimate = x.clone();
                }
                
                normgk = 0;
                for (int k = 0; k < gk.length; k++) {
                    normgk += gk[k] * gk[k];
                }
                normgk = Math.sqrt(normgk);
                System.out.println("1+");
                
                lk = (lk * 2 * sumAlphak + 2 * alpha_k * fk - alpha_k * alpha_k * normgk * normgk) / (2 * (sumAlphak + alpha_k));
                if (j == 1)
                    lk -= R * R / (2 * alpha_k);
                sumAlphak += alpha_k;
                lkbest = Math.max(lkbest, lk);
                
                for (int i = 0; i < x.length; i++) {
                    x[i] = Math.max(0, x[i] - alpha_k * gk[i]);
                }
                System.out.println("1+");

//                if (j%100 == 0 || j == 1)
                    System.out.println("iter =  " + j + ", fk = " + fk + " :: gradnorm = " + normgk + " :: fkbest = " + fkbest + " :: lkbest = " + lkbest);
            }
            System.out.println("iter =  " + j + ", fk = " + fk + " :: gradnorm = " + normgk + " :: fkbest = " + fkbest + " :: lkbest = " + lkbest);
        } catch (Exception exception) {
            throw new RuntimeException("Error while learning weights!", exception);
        }        
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = minEstimate[i];
        }
        System.out.println(Arrays.toString(weights));
        System.out.println("optimization complete.");
    }
    
    private class DisparityGradientComputer implements GradientCompute {

	private float B;
	
	DisparityGradientComputer(float B) {
	    this.B = B;
	}
	
        @Override
        public double computeFunctionGradient(double[] lambda, double[] grad) throws Exception {
            double obj = computeFirstGradient(lambda, grad);
            obj += computeSecondGradient(lambda, grad);
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
            double[][] C = new double[nu.length][nu.length];
            for (int j = 0; j < C.length; j++) {
                for (int k = 0; k < C.length; k++) {
                    for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
                        for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                            if (baseIndicator[j] == 1 && baseIndicator[k] == 1) {
                                double sum = 0;
                                for (int i = 0; i < baseIndicator.length; i++) {
                                    if (baseIndicator[i] == 1)
                                        sum += nu[i];
                                }
                                int gpIndex = groupIndex.get(new TIntArrayList(baseIndicator));
                                C[j][k] += weights[gpIndex] * nu[j] * nu[k] / (sum*sum);
                            }
                        }
                    }
                }
            }
            Matrix matrix = new Matrix(C);
            matrix = matrix.times(-1);
            double[] eigenValues = matrix.eig().getRealEigenvalues();
            double maxEig = -Double.MAX_VALUE;
            int index = -1;
            for (int i = 0; i < eigenValues.length; i++) {
                if (eigenValues[i] > 1e-10) {
                    System.out.println("weights = " + Arrays.toString(weights));
                    throw new IllegalStateException();
                }
                if (maxEig < eigenValues[i]) {
                    maxEig = eigenValues[i];
                    index = i;
                }
            }
            float[] eigenVector = new float[C.length];
            for (int i = 0; i < C.length; i++) {
                eigenVector[i] = (float) matrix.eig().getV().get(i, index);
            }

            for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
                for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                    double sum = 0;
                    for (int i = 0; i < baseIndicator.length; i++) {
                        if (baseIndicator[i] == 1)
                            sum += nu[i];
                    }
                    int gpIndex = groupIndex.get(new TIntArrayList(baseIndicator));
                    for (int j = 0; j < C.length; j++) {
                        for (int k = 0; k < C.length; k++) {
                            if (baseIndicator[j] == 1 && baseIndicator[k] == 1) {
                                coeffs[gpIndex] -=  B * eigenVector[j] * eigenVector[k] * nu[j] * nu[k] / (sum*sum);
                            }
                        }
                    }
                }
            }
            return B*maxEig;
        }
        
    }

    public double[] getEstimates() {
        return this.estimates;
    }

    public double getObjective() {
        return this.objective;
    }

    private double doBlockCoordinateGradientDescent(double[] x) throws Exception {
        MirrorDescentTrainer mirrorDescentTrainer = new MirrorDescentTrainer(0, 1, 1e-12);
        MMDConsistencyGradientCompute gradientCompute = new MMDConsistencyGradientCompute(A, b, nu, dataHandler);
        double[] prevx = new double[x.length];
        Arrays.fill(prevx, 1 / A.length);
        double prevObj;
        double currObj = Double.MAX_VALUE;
        int i = 0;
        do {
            prevObj = currObj;
            mirrorDescentTrainer.setMaxIter((int) Math.min(Math.pow(2, i), 256));
            System.arraycopy(x, 0, prevx, 0, x.length);
            System.out.print(i + " : ");
            System.out.print("[");
            for (int j = 0; j < b.length; j++) {
                if (j % 10 == 0)
                    System.out.print(j + " ... ");
                gradientCompute.set(x, j);
                int shift = j * A.length;
                double[] currTheta = new double[A.length];
                System.arraycopy(x, shift, currTheta, 0, currTheta.length);
                mirrorDescentTrainer.optimize(currTheta, gradientCompute);
                System.arraycopy(currTheta, 0, x, shift, currTheta.length);
            }
            System.out.print("] :: ");
            currObj = getMMDObjective(x);
            System.out.println(currObj);
            i++;
        } while (Math.abs(prevObj - currObj) > 1e-4 && i < 100);
        return getMMDObjective(x);
    }

    private double getMMDObjective(double[] x) {
        double sum = 0;
        for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                double weight = getWeight(baseIndicator, nu, scheme);
                double totalNu = 0;
                for (int i = 0; i < b.length; i++) {
                    if (baseIndicator[i] == 1)
                        totalNu += nu[i];
                }
                double[] thetahat = new double[A.length];
                double[] bg = new double[A.length];
                for (int i = 0; i < b.length; i++) {
                    int shift = i * A.length;
                    if (baseIndicator[i] == 1) {
                        for (int j = 0; j < A.length; j++) {
                            thetahat[j] = thetahat[j] + nu[i] * x[shift + j];
                            bg[j] = bg[j] + nu[i] * b[i][j];
                        }
                    }
                }
                for (int j = 0; j < A.length; j++) {
                    thetahat[j] = thetahat[j] / totalNu;
                    bg[j] = bg[j] / totalNu;
                }

                double[] Atheta = new double[A.length];
                for (int k = 0; k < A.length; k++) {
                    Atheta[k] = dot(A[k], thetahat);
                }
                double thetaAtheta = dot(thetahat, Atheta);
                double btheta = dot(bg, thetahat);
                sum = sum + weight * (thetaAtheta - 2 * btheta);
            }
        }
        return sum;
    }

    private static double[] add(double[] x, double[] y, double scale) {
        if (scale == 0)
            return x.clone();
        double[] z = new double[x.length];
        for (int i = 0; i < z.length; i++) {
            z[i] = x[i] + scale * y[i];
        }
        return z;
    }

    private static double dot(double[] x, double[] y) {
        double z = 0;
        for (int i = 0; i < x.length; i++) {
            z += x[i] * y[i];
        }
        return z;
    }

    private double getWeight(int[] baseIndicator, int[] nu, WeightScheme scheme) {
        switch (scheme) {
        case ROOTSCHEME: {
            boolean flag = true;
            for (int i = 0; i < baseIndicator.length; i++) {
                if (baseIndicator[i] == 0)
                    flag = false;
            }
            if (flag)
                return 1;
            else
                return 0;
        }
        case PROPWEIGHTSCHEME: {
            int total = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                if (baseIndicator[i] == 1)
                    total = total + nu[i];
            }
            return total;
        }
        case INVPROPWEIGHTSCHEME: {
            int total = 0;
            double sum = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                sum = sum + nu[i];
                if (baseIndicator[i] == 1)
                    total = total + nu[i];
            }
            return sum / total;
        }
        case PROPCHILDSCHEME: {
            int total = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                if (baseIndicator[i] == 1)
                    total = total + 1;
            }
            return total;
        }
        case PROPCHILDWEIGHTSCHEME: {
            int count = 0;
            double sum = 0;
            double total = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                total = total + nu[i];
                if (baseIndicator[i] == 1) {
                    count = count + 1;
                    sum = sum + nu[i];
                }
            }
            return count * sum / total;
        }
        case PROPCHILDINVWEIGHTSCHEME: {
            int count = 0;
            double sum = 0;
            double total = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                total = total + nu[i];
                if (baseIndicator[i] == 1) {
                    count = count + 1;
                    sum = sum + nu[i];
                }
            }
            return count * total / sum;
        }
        case DISPARITYSCHEME: {
//            double total = 0;
//            double sum = 0;
//            for (int i = 0; i < baseIndicator.length; i++) {
//                if (baseIndicator[i] == 1) {
//                    total = total + nu[i];
//                    sum += nu[i] * nu[i];
//                }
//            }
//            return sum / (total * total);
            return weights[groupIndex.get(new TIntArrayList(baseIndicator))];
        }
        case ALLEQUAL:
        default:
            return 1;
        }
    }

    protected class MMDConsistencyGradientCompute implements GradientCompute {

        private double[][] A;
        private double[][] b;
        private int[] nu;
        private DataHandler dataHandler;
        private int currentBucket;
        private double[] otherThetas;

        public MMDConsistencyGradientCompute(double[][] A, double[][] b, int[] nu, DataHandler dataHandler) {
            this.A = A;
            this.b = b;
            this.nu = nu;
            this.dataHandler = dataHandler;
        }

        public void set(double[] otherThetas, int currentBucket) {
            if (this.otherThetas == null)
                this.otherThetas = new double[otherThetas.length];
            System.arraycopy(otherThetas, 0, this.otherThetas, 0, otherThetas.length);
            this.currentBucket = currentBucket;
        }

        @Override
        public double computeFunctionGradient(double[] lambda, double[] grad) throws Exception {
            for (int i = 0; i < grad.length; i++)
                grad[i] = 0;
            for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
                double[] grad1 = new double[grad.length];
                for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                    double weight = getWeight(baseIndicator, nu, scheme);
                    if (weight != 0)
                        grad1 = add(grad1, getGradient(baseIndicator, lambda, otherThetas, this.currentBucket), weight);
                }
                for (int j = 0; j < grad.length; j++) {
                    grad[j] += grad1[j];
                }
            }

            double obj = 0;
            for (int i = 0; i < grad.length; i++)
                obj += grad[i] * lambda[i];
            return obj;
        }

        private double[] getGradient(int[] baseIndicator, double[] currentTheta, double[] otherThetas, int currentBucket) {
            double[] grad = new double[A.length];
            double totalNu = 0;
            for (int i = 0; i < b.length; i++) {
                if (baseIndicator[i] == 1)
                    totalNu += nu[i];
            }
            double[] bg = new double[A.length];
            for (int i = 0; i < b.length; i++) {
                if (baseIndicator[i] == 1) {
                    for (int j = 0; j < A.length; j++) {
                        bg[j] = bg[j] + nu[i] * b[i][j];
                    }
                }
            }
            for (int j = 0; j < A.length; j++) {
                bg[j] = bg[j] / totalNu;
            }

            for (int i = currentBucket; i <= currentBucket; i++) {
                if (baseIndicator[i] == 1) {
                    double[] thetai = currentTheta;
                    for (int j = 0; j < b.length; j++) {
                        int shift2 = j * A.length;
                        if (baseIndicator[j] == 1) {
                            double[] thetaj = new double[A.length];
                            if (i != j) {
                                System.arraycopy(otherThetas, shift2, thetaj, 0, A.length);
                            } else {
                                System.arraycopy(currentTheta, 0, thetaj, 0, A.length);
                            }
                            for (int k = 0; k < A.length; k++) {
                                grad[k] += (nu[i] * nu[j] * 2 * dot(A[k], thetai)) / (totalNu * totalNu);
                            }
                        }
                    }
                }
            }
            for (int i = currentBucket; i <= currentBucket; i++) {
                if (baseIndicator[i] == 1) {
                    for (int j = 0; j < A.length; j++) {
                        grad[j] += (-2.0 * bg[j] * nu[i]) / totalNu;
                    }
                }
            }

            return grad;
        }

    }

}
