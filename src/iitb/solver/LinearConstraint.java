package iitb.solver;

public class LinearConstraint implements IConstraint {

    public float[] coefficients;
    public float constant;

    public LinearConstraint(int numCoefficients) {
        this.coefficients = new float[numCoefficients];
    }

}
