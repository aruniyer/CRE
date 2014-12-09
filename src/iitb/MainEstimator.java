package iitb;

import iitb.cre.IEstimator;
import iitb.cre.MMDEstimator;
import iitb.data.IDataStore;
import iitb.kernel.GaussianFunction;
import iitb.kernel.IKernel;
import iitb.kernel.IKernelFunction;
import iitb.kernel.OnTheFlyKernel;

import java.util.Arrays;

public class MainEstimator {

    public static void main(String[] args) throws Exception {
        System.out.print("Loading training data ... ");
        IDataStore trainStore = Utility.readCSVFile("data/shuttle_train.csv", ",", true);
        System.out.println("[DONE]");
        System.out.print("Loading test data ... ");
        IDataStore testStore = Utility.readCSVFile("data/shuttle_test.csv", ",", true);
        System.out.println("[DONE]");
        System.out.print("Computing kernel matrices ... ");
        
        // Using the mean distance as the bandwidth parameter simply for testing purposes
        IKernelFunction kernelFunction = new GaussianFunction(Utility.meanDistance(trainStore));
        IKernel trainTrainKernel = new OnTheFlyKernel(trainStore, kernelFunction);
        IKernel trainTestKernel = new OnTheFlyKernel(trainStore, testStore, kernelFunction);
        System.out.println("[DONE]");
        System.out.println("Getting estimate ... ");
        IEstimator classRatioEstimator = new MMDEstimator(trainTrainKernel, trainTestKernel);
        System.out.println(Arrays.toString(classRatioEstimator.estimateFractions()));
        System.out.println("Actual proportions ... ");
        System.out.println(Arrays.toString(Utility.getClassProps(testStore)));
    }
    
}
