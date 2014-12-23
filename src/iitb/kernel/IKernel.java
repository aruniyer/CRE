package iitb.kernel;

import java.util.Iterator;
import java.util.Map.Entry;

import iitb.data.IDataStore;
import iitb.data.IInstance;

public interface IKernel extends Iterable<Entry<Entry<IInstance, IInstance>, Double>> {

    public double get(int i, int j);

    public int getNumRows();

    public int getNumColumns();

    public IDataStore getDataStore1();

    public IDataStore getDataStore2();

    public Iterator<Entry<Entry<IInstance, IInstance>, Double>> iterator();

}
