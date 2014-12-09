package iitb.kernel;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;

import gnu.trove.list.array.TFloatArrayList;
import iitb.data.IInstance;

public class WeightedKernelFunction<K extends IKernelFunction> implements IKernelFunction {

    private List<K> kernelFunctions;
    private TFloatArrayList weights;
    
    public WeightedKernelFunction() {
        this.kernelFunctions = new LinkedList<K>();
        this.weights = new TFloatArrayList();
    }
    
    public WeightedKernelFunction(K[] kernelFunctions) {
        this.kernelFunctions = Arrays.asList(kernelFunctions);
        this.weights = new TFloatArrayList();
        for (int i = 0; i < kernelFunctions.length; i++) {
            this.weights.add(1);
        }
    }

    public WeightedKernelFunction(K[] kernelFunctions, float[] weights) {
        this.kernelFunctions = Arrays.asList(kernelFunctions);
        this.setWeights(weights);
    }

    public void addKernelFunction(K kernelFunction, float weight) {
        kernelFunctions.add(kernelFunction);
        weights.add(weight);
    }

    public void setWeights(float[] weights) {
        if (weights != null)
            this.weights = new TFloatArrayList(weights);
    }

    public void setWeight(int i, float value) {
        this.weights.set(i, value);
    }

    @Override
    public double get(IInstance instance1, IInstance instance2) {
        double sum = 0;
        int i = 0;
        for (IKernelFunction kernelFunction : kernelFunctions) {
            sum += weights.get(i) * kernelFunction.get(instance1, instance2);
            i++;
        }
        return sum;
    }

    @Override
    public void setParameters(Properties properties) {
        throw new UnsupportedOperationException("No parameter clauses for weighted kernels!");
    }

}
