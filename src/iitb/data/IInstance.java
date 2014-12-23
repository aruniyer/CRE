package iitb.data;

public interface IInstance extends Cloneable {

    public int numFeatures();

    public float get(int attributeIndex);

    public float label();
    
    public IInstance clone();

}
