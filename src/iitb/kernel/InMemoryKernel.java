package iitb.kernel;

import java.util.Iterator;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

import iitb.data.IDataStore;
import iitb.data.IInstance;

public class InMemoryKernel implements IKernel {

    private double[][] kernel;
    private IDataStore dataStore1;
    private IDataStore dataStore2;

    public InMemoryKernel(IDataStore dataStore, IKernelFunction kernelFunction) {
        this(dataStore, dataStore, kernelFunction);
    }

    public InMemoryKernel(IDataStore dataStore1, IDataStore dataStore2, IKernelFunction kernelFunction) {
        int rows = dataStore1.size();
        int cols = dataStore2.size();
        this.kernel = new double[rows][cols];
        int i = 0;
        for (IInstance instance1 : dataStore1) {
            int j = 0;
            for (IInstance instance2 : dataStore2) {
                this.kernel[i][j] = kernelFunction.get(instance1, instance2);
                j++;
            }
            i++;
        }
        this.dataStore1 = dataStore1;
        this.dataStore2 = dataStore2;
    }

    @Override
    public double get(int i, int j) {
        return kernel[i][j];
    }

    @Override
    public int getNumRows() {
        return kernel.length;
    }

    @Override
    public int getNumColumns() {
        return kernel[0].length;
    }

    @Override
    public IDataStore getDataStore1() {
        return this.dataStore1;
    }

    @Override
    public IDataStore getDataStore2() {
        return this.dataStore2;
    }

    @Override
    public Iterator<Entry<Entry<IInstance, IInstance>, Double>> iterator() {
        return new ArrayIterator();
    }

    private class ArrayIterator implements Iterator<Entry<Entry<IInstance, IInstance>, Double>> {

        private int i = 0;
        private int j = 0;

        @Override
        public boolean hasNext() {
            return i < getNumRows();
        }

        @Override
        public Entry<Entry<IInstance, IInstance>, Double> next() {
            double value = kernel[i][j];
            IInstance xi = getDataStore1().get(i);
            IInstance xj = getDataStore1().get(j);
            Entry<Entry<IInstance, IInstance>, Double> retVal = new SimpleEntry<Entry<IInstance, IInstance>, Double>(new SimpleEntry<IInstance, IInstance>(xi, xj), value);
            j++;
            if (j == getNumColumns()) {
                j = 0;
                i++;
            }
            return retVal;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Cannot remove entries from in memory kernel!");
        }

    }

}
