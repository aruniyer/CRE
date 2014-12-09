package iitb.cre;

import java.util.Map.Entry;

import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.kernel.IKernel;
import iitb.solver.AbstractQPSimplex;
import iitb.solver.QPSimplexMirrorDescent;

/**
 * 
 * Class Ratio Estimator from ICML 2014
 * 
 * @author aruniyer
 *
 */
public class MMDEstimator implements IEstimator {

    private double[][] A;
    private double[] b;
    AbstractQPSimplex solver;

    public MMDEstimator(double[][] A, double[] b) {
        setA(A);
        setb(b);
        this.solver = new QPSimplexMirrorDescent(this.A, this.b);
    }

    public MMDEstimator(IKernel trainTrainKernel, IKernel trainTestKernel) {
        setA(trainTrainKernel);
        setb(trainTestKernel);
        this.solver = new QPSimplexMirrorDescent();
    }

    public void setA(IKernel trainTrainKernel) {
        IDataStore train = trainTrainKernel.getDataStore1();
        int[] ny = new int[train.numClasses()];
        for (IInstance instance : train) {
            ny[(int) instance.label()]++;
        }
        this.A = new double[train.numClasses()][train.numClasses()];
        for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTrainKernel) {
            IInstance instance1 = entry.getKey().getKey();
            IInstance instance2 = entry.getKey().getValue();
            Double kernelValue = entry.getValue();
            int y1 = (int) instance1.label();
            int y2 = (int) instance2.label();
            this.A[y1][y2] += 2.0 * kernelValue / ((double) ny[y1] * ny[y2]);
        }
    }

    public void setA(double[][] A) {
        this.A = A;
    }

    public void setb(IKernel trainTestKernel) {
        IDataStore train = trainTestKernel.getDataStore1();
        int[] ny = new int[train.numClasses()];
        for (IInstance instance : train) {
            ny[(int) instance.label()]++;
        }
        IDataStore test = trainTestKernel.getDataStore2();
        int nu = test.size();
        this.b = new double[train.numClasses()];
        for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTestKernel) {
            IInstance instance1 = entry.getKey().getKey();
            Double kernelValue = entry.getValue();
            int y = (int) instance1.label();
            this.b[y] -= 2.0 * kernelValue / ((double) ny[y] * nu);
        }
    }

    public void setb(double[] b) {
        this.b = b;
    }

    @Override
    public double[] estimateFractions() {
        this.solver.setA(A);
        this.solver.setb(b);
        this.solver.solve();
        return this.solver.getSolution();
    }
    
    public double getObjective() {
        return this.solver.getObjective();
    }

}
