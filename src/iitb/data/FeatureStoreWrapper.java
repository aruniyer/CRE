package iitb.data;

import java.util.Iterator;

import iitb.shared.repository.fixedLengthRecords.FeatureStore;
import iitb.shared.repository.fixedLengthRecords.FeatureStore.IteratorWithCount;
import iitb.shared.repository.fixedLengthRecords.Instance;

public class FeatureStoreWrapper implements IDataStore {

    private FeatureStore featureStore;

    public FeatureStoreWrapper(FeatureStore featureStore) {
        this.featureStore = featureStore;
    }

    @Override
    public Iterator<IInstance> iterator() {
        try {
            return new IteratorWrapper(featureStore.allInstances());
        } catch (Exception exception) {
            throw new RuntimeException("Exception in feature store!", exception);
        }
    }

    @Override
    public int size() {
        return featureStore.size();
    }

    private class IteratorWrapper implements Iterator<IInstance> {

        private IteratorWithCount iterator;

        public IteratorWrapper(IteratorWithCount iterator) {
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
            return instance.floatValue(attributeIndex);
        }

        @Override
        public float label() {
            return instance.trueClass();
        }

        @Override
        public int numFeatures() {
            return instance.numFeatures();
        }
        
        @Override
        public IInstance clone() {
            try {
                return (IInstance) super.clone();
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
            return null;
        }

    }

    @Override
    public int numClasses() {
        return featureStore.getFeatureList().numClasses();
    }
    
    @Override
    public int numFeatures() {
        return featureStore.getFeatureList().numFeatures();
    }

    @Override
    public IInstance get(int index) {
	try {
	    return new InstanceWrapper(featureStore.instancesAt(new int[]{index}).next());
	} catch (Exception exception) {
	    throw new RuntimeException("Unable to fetch instances from feature store!", exception);
	}
    }
    
    @Override
    public IDataStore getView(int[] indices) {
        SimpleDataStore viewStore = new SimpleDataStore(numClasses(), numFeatures());
        try {
            IteratorWithCount iterator = featureStore.instancesAt(indices);
            while (iterator.hasNext()) {
        	viewStore.add(new InstanceWrapper(iterator.next()));
            }
            return viewStore;
        } catch (Exception exception) {
            throw new RuntimeException("Unable to fetch instances from feature store!", exception);
        }
    }

}
