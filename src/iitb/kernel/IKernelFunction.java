package iitb.kernel;

import java.util.Properties;

import iitb.data.IInstance;

public interface IKernelFunction {
    
    public void setParameters(Properties properties);

    public double get(IInstance instance1, IInstance instance2);

}
