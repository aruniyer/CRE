package iitb.data;

public class SimpleInstance implements IInstance {

    private float[] vals;
    private float label;
    
    public SimpleInstance(int numValues) {
        vals = new float[numValues];
    }
    
    public SimpleInstance(float[] values, float label) {
        this.vals = values.clone();
        this.label = label;
    }
    
    public void set(int index, float value) {
        vals[index] = value;
    }
    
    public void set(float label) {
        this.label = label;        
    }
    
    @Override
    public int numFeatures() {
        return vals.length;
    }

    @Override
    public float get(int attributeIndex) {
        return vals[attributeIndex];
    }

    @Override
    public float label() {
        return label;
    }
    
    @Override
    public IInstance clone() {
        return (IInstance) this.clone();
    }

}
