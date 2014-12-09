package iitb.cre;

import java.util.Map.Entry;

import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.kernel.IKernel;
import iitb.kernel.IKernelFunction;
import iitb.kernel.OnTheFlyKernel;

public abstract class AbstractKernelLearning {

    protected IDataStore D;
    protected IDataStore[] Us;
    protected IKernelFunction[] kernelFunctions;
    protected float[][][] A;
    protected float[][][] b;
    protected float[] weights;
    protected float C1, C2;

    public AbstractKernelLearning(IDataStore D, IDataStore[] Us, IKernelFunction[] kernelFunctions, float C1, float C2) {
        this.D = D;
        this.Us = Us;
        this.C1 = C1;
        this.C2 = C2;
        this.kernelFunctions = kernelFunctions;
        int numY = D.numClasses();
        System.out.println("Loading all A's for the training ... ");
        A = new float[kernelFunctions.length][numY][numY];
        for (int i = 0; i < kernelFunctions.length; i++) {
            setA(new OnTheFlyKernel(D, kernelFunctions[i]), A[i]);
        }
        System.out.println("Loading all b's for each kernel and U ... ");
        b = new float[Us.length][kernelFunctions.length][numY];
        for (int j = 0; j < Us.length; j++) {
            for (int i = 0; i < kernelFunctions.length; i++) {
                setb(new OnTheFlyKernel(D, Us[j], kernelFunctions[i]), b[j][i]);
            }
        }
    }

    private void setA(IKernel trainTrainKernel, float[][] A) {
        IDataStore train = trainTrainKernel.getDataStore1();
        int[] ny = new int[train.numClasses()];
        for (IInstance instance : train) {
            ny[(int) instance.label()]++;
        }

        for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTrainKernel) {
            IInstance instance1 = entry.getKey().getKey();
            IInstance instance2 = entry.getKey().getValue();
            Double kernelValue = entry.getValue();
            int y1 = (int) instance1.label();
            int y2 = (int) instance2.label();
            A[y1][y2] += 2.0 * kernelValue / ((double) ny[y1] * ny[y2]);
        }
    }

    private void setb(IKernel trainTestKernel, float[] b) {
        IDataStore train = trainTestKernel.getDataStore1();
        int[] ny = new int[train.numClasses()];
        for (IInstance instance : train) {
            ny[(int) instance.label()]++;
        }
        IDataStore test = trainTestKernel.getDataStore2();
        int nu = test.size();
        for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTestKernel) {
            IInstance instance1 = entry.getKey().getKey();
            Double kernelValue = entry.getValue();
            int y = (int) instance1.label();
            b[y] -= 2.0 * kernelValue / ((double) ny[y] * nu);
        }
    }

    public abstract void learnWeights();

    public float[] getWeights() {
        return weights;
    }

    /*
     * Computes x^TAx + bx
     */
    protected float getQObj(float[][] A, float[] b, float[] x) {
        float sum = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A.length; j++) {
                sum += x[i] * x[j] * A[i][j];
            }
            sum += b[i] * x[i];
        }
        return sum;
    }

}