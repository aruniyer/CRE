package iitb.kernel;

import java.util.Iterator;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

import iitb.data.IDataStore;
import iitb.data.IInstance;

public class OnTheFlyKernel implements IKernel {

    private IDataStore dataStore1;
    private IDataStore dataStore2;
    private IKernelFunction kernelFunction;

    public OnTheFlyKernel(IDataStore dataStore, IKernelFunction kernelFunction) {
        this(dataStore, dataStore, kernelFunction);
    }

    public OnTheFlyKernel(IDataStore dataStore1, IDataStore dataStore2, IKernelFunction kernelFunction) {
        this.dataStore1 = dataStore1;
        this.dataStore2 = dataStore2;
        this.kernelFunction = kernelFunction;
    }

    @Override
    public double get(int i, int j) {
        return this.kernelFunction.get(dataStore1.get(i), dataStore2.get(j));
    }

    @Override
    public int getNumRows() {
        return dataStore1.size();
    }

    @Override
    public int getNumColumns() {
        return dataStore2.size();
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
        return new DataIterator();
    }

    private class DataIterator implements Iterator<Entry<Entry<IInstance, IInstance>, Double>> {

        private Iterator<IInstance> iterator1;
        private Iterator<IInstance> iterator2;
        private IInstance instance1;

        public DataIterator() {
            iterator1 = getDataStore1().iterator();
            iterator2 = getDataStore2().iterator();
            instance1 = iterator1.next();
        }

        @Override
        public boolean hasNext() {
            return iterator1.hasNext();
        }

        @Override
        public Entry<Entry<IInstance, IInstance>, Double> next() {
            Entry<Entry<IInstance, IInstance>, Double> retVal = null;
            if (!iterator2.hasNext()) {
                iterator2 = getDataStore2().iterator();
                instance1 = iterator1.next();
            }
            IInstance instance2 = iterator2.next();
            double value = kernelFunction.get(instance1, instance2);
            retVal = new SimpleEntry<Entry<IInstance, IInstance>, Double>(new SimpleEntry<IInstance, IInstance>(instance1, instance2), value);
            return retVal;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Cannot remove entries from On the Fly Kernel!");
        }

    }

}
