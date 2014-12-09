package iitb.data;

public interface IDataStore extends Iterable<IInstance> {

    public IInstance get(int index);

    public int size();

    public int numClasses();
    
    public int numFeatures();
    
    public IDataStore getView(int[] indices);

}
