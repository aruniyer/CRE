package iitb.cre.confint;

import java.util.Arrays;

import org.apache.commons.math3.special.Gamma;

import iitb.Utility;
import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.shared.optimization.GradientCompute;

public class BetaVarianceModel {

    private IDataStore D;
    private double[] weights;

    public BetaVarianceModel(IDataStore D) {
        this.D = D;
    }

    public void learn() {
        weights = new double[D.numFeatures()];
        Arrays.fill(weights, 1.0);
        try {
            GradientCompute gc = new GradientComputer();
            double[] grad = new double[weights.length];
            double normgrad = 0;
            int iter = 0;
            double prevObj = -Double.MAX_VALUE;
            double alpha = 1;
            do {
                iter++;
                double obj = gc.computeFunctionGradient(weights, grad);
                normgrad = 0;
                for (int i = 0; i < grad.length; i++)
                    normgrad += grad[i]*grad[i];
                normgrad = Math.sqrt(normgrad);
                
                if (prevObj < obj) {
                    for (int i = 0; i < grad.length; i++) {
                        weights[i] += alpha * (grad[i] / normgrad);
                    }
                    prevObj = obj;
                } else {
                    alpha = 1 / iter;
                }
                
                System.out.println("Obj = " + prevObj + " :: normgrad = " + normgrad);
            } while (normgrad > 1e-4 && iter < 1000);
        } catch (Exception exception) {
            throw new RuntimeException("Error while learning weights!", exception);
        }
    }

    public double[] getWeights() {
        return this.weights;
    }

    public double predict(IInstance instance) {
        double sum = weights[0];
        for (int i = 1; i < instance.numFeatures(); i++) {
            sum += weights[i] * instance.get(i);
        }
        return ginv(sum);
    }
    
    private double ginv(double in) {
        return in * in;
    }

    private double dginv(double in) {
        return 2 * in;
    }

    private class GradientComputer implements GradientCompute {

        @Override
        public double computeFunctionGradient(double[] w, double[] grad) throws Exception {
            Arrays.fill(grad, 0);
            for (IInstance instance : D) {
                instanceGradient(instance, w, grad);
            }
            return getObjective(w);
        }
        
        private double getObjective(double[] w) {
            double obj = 0;
            for (IInstance instance : D) {
                double muM = instance.get(0);
                double[] xi = new double[w.length];
                xi[0] = 1;
                for (int i = 1; i < instance.numFeatures(); i++) {
                    xi[i] = instance.get(i);
                    if (i == 1)
                        xi[i] = ginv(xi[i]);
                }
                double yi = instance.label();

                double wtx = 0;
                for (int i = 1; i < xi.length; i++) {
                    wtx += w[i] * xi[i];
                }

                double mi = muM * (ginv(wtx) - 2) + 1;
                double ni = (1 - muM) * ginv(wtx) + 2 * muM;
                
                obj += (mi - 1) * Math.log(yi);
                obj += (ni - 1) * Math.log(1 - yi);
                obj -= Gamma.logGamma(mi);
                obj -= Gamma.logGamma(ni);
                obj += Gamma.logGamma(mi + ni);
            }
            return obj;
        }
        
        private void instanceGradient(IInstance instance, double[] w, double[] grad) {
            double muM = instance.get(0);
            double[] xi = new double[grad.length];
            xi[0] = 1;
            for (int i = 1; i < instance.numFeatures(); i++) {
                xi[i] = instance.get(i);
            }
            double yi = instance.label();

            double wtx = 0;
            for (int i = 1; i < xi.length; i++) {
                wtx += w[i] * xi[i];
            }

            double mi = muM * (ginv(wtx) - 2) + 1;
            double ni = (1 - muM) * ginv(wtx) + 2 * muM;
            for (int i = 0; i < grad.length; i++) {
                double dmidw = (muM * dginv(wtx) * xi[i]);
                double dnidw = ((1 - muM) * dginv(wtx) * xi[i]);
                grad[i] += mi * Math.log(yi) * dmidw;
                grad[i] += ni * Math.log(1 - yi) * dnidw;
                grad[i] += -Gamma.digamma(mi) * dmidw;
                grad[i] += -Gamma.digamma(ni) * dnidw;
                grad[i] += Gamma.digamma(mi + ni) * (dmidw + dnidw);
            }
        }

    }

    public static void main(String[] args) throws Exception {
        IDataStore D = Utility.readCSVFile("australian_beta_features", ",", true, true);
        BetaVarianceModel model = new BetaVarianceModel(D);
        model.learn();
        for (IInstance instance : D) {
            System.out.println(instance.get(1) + " :: " + Math.sqrt(model.predict(instance)));
        }
        System.out.println(Arrays.toString(model.getWeights()));
    }

}
