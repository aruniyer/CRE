package iitb.data;

import java.util.Iterator;

import gnu.trove.map.hash.TIntObjectHashMap;

public class SimpleDataStore implements IDataStore {
    
    private TIntObjectHashMap<IInstance> instanceMap;
    private int numClasses, numFeatures;

    public SimpleDataStore(int numClasses, int numFeatures) {
        this.instanceMap = new TIntObjectHashMap<IInstance>();
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
    }
    
    public void add(IInstance instance) {
        int size = this.size();
        this.instanceMap.put(size, instance);
    }
    
    @Override
    public Iterator<IInstance> iterator() {
        return instanceMap.valueCollection().iterator();
    }

    @Override
    public IInstance get(int index) {
        return instanceMap.get(index);
    }

    @Override
    public int size() {
        return instanceMap.size();
    }

    @Override
    public int numClasses() {
        return this.numClasses;
    }

    @Override
    public int numFeatures() {
        return this.numFeatures;
    }

    @Override
    public IDataStore getView(int[] indices) {
        SimpleDataStore viewStore = new SimpleDataStore(numClasses, numFeatures);
        for (int index : indices) {
            viewStore.add(this.get(index));
        }
        return viewStore;
    } 

}