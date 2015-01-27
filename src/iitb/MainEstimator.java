package iitb;

import iitb.cre.IEstimator;
import iitb.cre.MMDEstimator;
import iitb.data.IDataStore;
import iitb.kernel.GaussianFunction;
import iitb.kernel.IKernel;
import iitb.kernel.IKernelFunction;
import iitb.kernel.OnTheFlyKernel;
import iitb.util.Utility;

import java.util.Arrays;

public class MainEstimator {

    public static void main(String[] args) throws Exception {
        String prefix = "data/";
        String trainingFile = prefix + "shuttle_train.csv";
        String testingFile = prefix + "shuttle_test.csv";
        String trainSeparator = ",";
        String testSeparator = ",";
        boolean headerTrain = true;
        boolean headerTest = true;
        
        // kernel file indicates the file with kernel weights learned
        // if set to null, the estimator uses a default kernel with 
        // mean distance as bandwidth
        String kernelFile = "shuttle.klist";
        doEstimation(trainingFile, trainSeparator, headerTrain, testingFile, testSeparator, headerTest, kernelFile);        
    }
    
    public static void doEstimation(String trainingFile, String trainSeparator, boolean headerTrain, String testingFile, String testSeparator, boolean headerTest, String kernelFile) throws Exception {
        System.out.print("Loading training data ... ");
        IDataStore trainStore = Utility.readCSVFile(trainingFile, trainSeparator, headerTrain);
        System.out.println("[DONE]");
        System.out.print("Loading test data ... ");
        IDataStore testStore = Utility.readCSVFile(testingFile, testSeparator, headerTest);
        System.out.println("[DONE]");
        System.out.print("Computing kernel matrices ... ");
        IKernelFunction kernelFunction;
        if (kernelFile == null)
            kernelFunction = new GaussianFunction(Utility.meanDistance(trainStore));
        else
            kernelFunction = Utility.readKernelFile(kernelFile);
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
