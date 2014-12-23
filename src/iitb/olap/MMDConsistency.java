package iitb.olap;

import java.util.Arrays;

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
        double[] x = new double[b.length * A.length];
        Arrays.fill(x, 1.0 / A.length);
        this.objective = doBlockCoordinateGradientDescent(x);
        this.estimates = x;
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
            double total = 0;
            double sum = 0;
            for (int i = 0; i < baseIndicator.length; i++) {
                if (baseIndicator[i] == 1) {
                    total = total + nu[i];
                    sum += nu[i] * nu[i];
                }
            }
            return Math.sqrt((total * total) / sum);
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
