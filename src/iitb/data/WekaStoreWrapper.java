package iitb.data;

import java.util.Iterator;

import weka.core.Instance;
import weka.core.Instances;

public class WekaStoreWrapper implements IDataStore {

    private Instances instances;

    public WekaStoreWrapper(Instances instances) {
        this.instances = instances;
    }

    @Override
    public Iterator<IInstance> iterator() {
        return new IteratorWrapper(instances.iterator());
    }

    @Override
    public int size() {
        return instances.size();
    }

    private class IteratorWrapper implements Iterator<IInstance> {

        private Iterator<Instance> iterator;

        public IteratorWrapper(Iterator<Instance> iterator) {
            this.iterator = iterator;
        }

        @Override
        public boolean hasNext() {
            return iterator.hasNext();
        }

        @Override
        public IInstance next() {
            return new InstanceWrapper(iterator.next());
        }

        @Override
        public void remove() {
            iterator.remove();
        }

    }

    private class InstanceWrapper implements IInstance {

        private Instance instance;

        public InstanceWrapper(Instance instance) {
            this.instance = instance;
        }

        @Override
        public float get(int attributeIndex) {
            return (float) instance.value(attributeIndex);
        }

        @Override
        public float label() {
            return (float) instance.classValue();
        }

        @Override
        public int numFeatures() {
            return instance.numAttributes() - 1;
        }
        
        @Override
        public IInstance clone() {
            return this.clone();
        }

    }

    @Override
    public int numClasses() {
        return instances.numClasses();
    }

    @Override
    public int numFeatures() {
        return instances.numAttributes() - 1;
    }

    @Override
    public IInstance get(int index) {
        return new InstanceWrapper(instances.get(index));
    }
    
    @Override
    public IDataStore getView(int[] indices) {
        Instances freshView = new Instances(instances, indices.length);
        for (int index : indices) {
            freshView.add(instances.get(index));
        }
        return new WekaStoreWrapper(freshView);
    }

}
