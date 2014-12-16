package iitb.kernel;

import java.util.Properties;

import iitb.data.IInstance;

public class GaussianFunction implements IKernelFunction {

    public static String SIGMA = "sigma";
    public static String RANGED = "ranged";
    public static String ATTRIBUTES = "attributes";
    
    private double sigma;
    private double gamma;
    private boolean ranged;
    private int[] attributes = null;
    
    public GaussianFunction() {
        this(1, false, null);
    }

    public GaussianFunction(double sigma) {
        this(sigma, false, null);
    }
    
    public GaussianFunction(double sigma, int attribute) {
        this(sigma, true, new int[]{attribute});
    }

    public GaussianFunction(double sigma, boolean ranged, int[] attributes) {
        this.sigma = sigma;
        this.gamma = 1.0 / (2.0 * sigma * sigma);
        if (ranged) {
            this.ranged = true;
            this.attributes = attributes;
        }
    }
    
    public double getSigma() {
        return this.sigma;
    }
    
    public boolean isRanged() {
        return this.ranged;
    }
    
    public int[] getAttributes() {
        return this.attributes;
    }

    public double get(IInstance instance1, IInstance instance2) {
        double sum = 0;
        if (ranged) {
            for (int i : attributes) {
                double val1 = instance1.get(i);
                double val2 = instance2.get(i);
                sum += (val1 - val2) * (val1 - val2);
            }
        } else {
            for (int i = 0; i < instance1.numFeatures(); i++) {
                double val1 = instance1.get(i);
                double val2 = instance2.get(i);
                sum += (val1 - val2) * (val1 - val2);
            }
        }
        return Math.exp(-gamma * sum);
    }

    @Override
    public void setParameters(Properties properties) {
        String sigma = properties.getProperty(SIGMA);
        if (sigma != null) {
            this.sigma = Double.valueOf(sigma);
        }
        
        String ranged = properties.getProperty(RANGED);
        if (ranged != null) {
            this.ranged = Boolean.valueOf(ranged);
        }
        
        if (this.ranged) {
            String attributes = properties.getProperty(ATTRIBUTES);
            String[] atts = attributes.split(",");
            this.attributes = new int[atts.length];
            for (int i = 0; i < atts.length; i++) {
                this.attributes[i] = Integer.parseInt(atts[i]);
            }
        }
    }

}
