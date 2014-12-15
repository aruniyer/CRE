package iitb;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;

import iitb.cre.IEstimator;
import iitb.cre.KernelLearningGD;
import iitb.cre.MMDEstimator;
import iitb.data.IDataStore;
import iitb.kernel.GaussianFunction;
import iitb.kernel.IKernel;
import iitb.kernel.OnTheFlyKernel;
import iitb.kernel.WeightedKernelFunction;

public class MainLearning {
    
    public static void main(String[] args) throws Exception {
        String prefix = "data/";
        String trainingFile = prefix + "shuttle_train.csv";
        String outputFile = "output";
        String testingFile = prefix + "shuttle_test.csv";
        String trainSeparator = ",";
        String testSeparator = ",";
        boolean headerTrain = true;
        boolean headerTest = true;
        
        System.out.print("Loading training data ... ");
        IDataStore D = Utility.readCSVFile(trainingFile, trainSeparator, headerTrain);
        System.out.println("[DONE]");
        System.out.print("Selecting bandwidth ... ");
        double bandwidth = Utility.selectBandwidth(D);
        System.out.println("[DONE] (" + bandwidth + ")");
        System.out.println("Running kernel learning procedure ... ");
        int numFeatures = D.numFeatures();
        double[] log2bs = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
        GaussianFunction[] baseKernelFunctions = new GaussianFunction[log2bs.length];
        for (int i = 0; i < log2bs.length; i++) {
            baseKernelFunctions[i] = new GaussianFunction(Math.pow(2, log2bs[i]) * bandwidth * numFeatures * numFeatures);
        }
        int numProportions = 11;
        int numPrototypesPerProportion = 2;
        int sizePerPrototype = 20;
        long seed = 1;
        IDataStore[] Us = Utility.getPrototypes(D, seed, numProportions, numPrototypesPerProportion, sizePerPrototype);
        IDataStore[] validate = Utility.getPrototypes(D, 20, numProportions, numPrototypesPerProportion, sizePerPrototype);
        double minValidationError = Double.MAX_VALUE;
        float[] minWeights = null;
        float minC1 = -1, minC2 = -1;
        for (float C1 : new float[]{0, 1, 10, 100, 1000}) {
            for (float C2 : new float[]{0, 1, 10, 100, 1000}) {
                if (C1 == 0 && C2 == 0)
                    continue;
                System.out.println("C1 = " + C1 + ", C2 = " + C2);
                KernelLearningGD kernelLearning = new KernelLearningGD(D, Us, baseKernelFunctions, C1, C2);
                kernelLearning.learnWeights();
                System.out.println("kernel learning procedure completed.");
                
                System.out.print("Base Kernel Sigmas : ");
                for (GaussianFunction gaussianFunction : baseKernelFunctions) {
                    System.out.print(gaussianFunction.getSigma() + " ");
                }
                System.out.println();
                float[] weights = kernelLearning.getWeights();
                System.out.println("Weights Learned = " + Arrays.toString(weights));
                
                WeightedKernelFunction<GaussianFunction> weightedKernelFunction = new WeightedKernelFunction<GaussianFunction>(baseKernelFunctions);
                weightedKernelFunction.setWeights(weights);
                IKernel trainTrainKernel = new OnTheFlyKernel(D, weightedKernelFunction);
                double validationError = 0;
                for (int i = 0; i < validate.length; i++) {
                    IKernel trainTestKernel = new OnTheFlyKernel(D, validate[i], weightedKernelFunction);
                    IEstimator classRatioEstimator = new MMDEstimator(trainTrainKernel, trainTestKernel);
                    double[] estimate = classRatioEstimator.estimateFractions();
                    double[] props = Utility.getClassProps(validate[i]);
                    for (int j = 0; j < estimate.length; j++) {
                        double diff = (estimate[j] - props[j]);
                        validationError += diff*diff;
                    }
                }
                validationError /= validate.length;
                if (validationError < minValidationError) {
                    minValidationError = validationError;
                    minWeights = weights;
                    minC1 = C1;
                    minC2 = C2;
                }
            }
        }
        
        if (outputFile != null && !outputFile.trim().isEmpty()) {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile));
            for (int i = 0; i < baseKernelFunctions.length; i++) {
                bufferedWriter.write(baseKernelFunctions[i].getClass().getName());
                bufferedWriter.write("#" + GaussianFunction.SIGMA + "=" + baseKernelFunctions[i].getSigma());
                bufferedWriter.write("#" + minWeights[i]);
                bufferedWriter.newLine();
            }
            bufferedWriter.close();
        }
        
        if (testingFile != null && !testingFile.trim().isEmpty()) {
            System.out.print("Loading test data ... ");
            IDataStore U = Utility.readCSVFile(testingFile, testSeparator, headerTest);
            System.out.println("[DONE]");
            WeightedKernelFunction<GaussianFunction> weightedKernelFunction = new WeightedKernelFunction<GaussianFunction>(baseKernelFunctions);
            weightedKernelFunction.setWeights(minWeights);
            System.out.print("Computing kernel matrices ... ");
            IKernel trainTrainKernel = new OnTheFlyKernel(D, weightedKernelFunction);
            IKernel trainTestKernel = new OnTheFlyKernel(D, U, weightedKernelFunction);
            System.out.println("[DONE]");
            System.out.println("Getting estimate ... ");
            IEstimator classRatioEstimator = new MMDEstimator(trainTrainKernel, trainTestKernel);
            System.out.println(Arrays.toString(classRatioEstimator.estimateFractions()));
            System.out.println("Actual proportions ... ");
            System.out.println(Arrays.toString(Utility.getClassProps(U)));
        }
        System.out.println(Arrays.toString(minWeights));
        System.out.println(minC1 + ", " + minC2);
    }

}
