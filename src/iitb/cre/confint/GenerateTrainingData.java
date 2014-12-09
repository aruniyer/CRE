package iitb.cre.confint;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Map.Entry;

import org.apache.commons.math3.analysis.solvers.LaguerreSolver;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import gnu.trove.list.array.TIntArrayList;
import iitb.Utility;
import iitb.cre.MMDEstimator;
import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.data.SimpleDataStore;
import iitb.data.SimpleInstance;
import iitb.kernel.GaussianFunction;
import iitb.kernel.IKernel;
import iitb.kernel.IKernelFunction;
import iitb.kernel.OnTheFlyKernel;

public class GenerateTrainingData {

    private static int FEATURES_TO_COMPUTE = 7;

    public static void main(String[] args) throws Exception {
        String prefix = "/home/aruniyer/Workspace/Data/forpedrtest/";
        String dataFile = prefix + "Australian_train_0.1_1_400";
        String kernelInformationFile = "australiankernels";
        String outputFile = "australian_beta_features";
        
        System.out.print("Loading training data ... ");
        IDataStore D = Utility.readCSVFile(dataFile, " ", false);
        System.out.println("[DONE]");
        System.out.print("Loading the kernel functions ... ");
        IKernelFunction kernelFunction;
        if (kernelInformationFile != null && !kernelInformationFile.isEmpty())
            kernelFunction = Utility.readKernelFile(kernelInformationFile);
        else
            kernelFunction = new GaussianFunction(Utility.selectBandwidth(D));
        System.out.println("[DONE]");
        GenerateTrainingData generateTrainingData = new GenerateTrainingData();
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile));
        for (int i = 0; i < FEATURES_TO_COMPUTE + 1; i++) {
            bufferedWriter.write("f" + i + ",");
        }
        bufferedWriter.write("label");
        bufferedWriter.newLine();
        for (int size : new int[]{50, 100, 150, 200, 250, 300}) {
            System.out.println("Generating training data for size " + size + " ... ");
            IDataStore trainingData = generateTrainingData.generateTrainingData(D, kernelFunction, size, 1);
            for (IInstance instance : trainingData) {
                for (int i = 0; i < instance.numFeatures(); i++) {
                    bufferedWriter.write(instance.get(i) + ",");
                }
                bufferedWriter.write(instance.label() + "");
                bufferedWriter.newLine();
            }
            bufferedWriter.flush();
        }
        bufferedWriter.close();
    }

    public IDataStore generateTrainingData(IDataStore D, IKernelFunction kernelFunction, int sizePerPrototype, long seed) {
        int numProportions = 11;
        int numPrototypesPerProportion = 1;
        IDataStore[] Us = Utility.getPrototypes(D, seed, numProportions, numPrototypesPerProportion, sizePerPrototype);
        SimpleDataStore simpleDataStore = new SimpleDataStore(2, FEATURES_TO_COMPUTE);
        for (int i = 0; i < Us.length; i++) {
            System.out.println("U = " + (i + 1) + "/" + Us.length);
            IDataStore Ui = Us[i];
            double[] props = Utility.getClassProps(Ui);
            float[] features = computeFeatures(D, Ui, kernelFunction);
            simpleDataStore.add(new SimpleInstance(features, (float) props[1]));
        }
        return simpleDataStore;
    }

    public float[] computeFeatures(IDataStore D, IDataStore Ui, IKernelFunction kernelFunction) {
        int k = 10;
        TIntArrayList[] splitsD = Utility.splitter(D, k, 1);
        TIntArrayList[] splitsU = Utility.splitter(Ui, k, 1);

        int[] valD = new int[D.size()];
        for (int i = 0; i < D.size(); i++) {
            valD[i] = i;
        }

        int[] valU = new int[Ui.size()];
        for (int i = 0; i < Ui.size(); i++) {
            valU[i] = i;
        }

        float[] features = new float[FEATURES_TO_COMPUTE + 1];
        DescriptiveStatistics objectiveValues = new DescriptiveStatistics();
        DescriptiveStatistics estimates = new DescriptiveStatistics();
        DescriptiveStatistics numerators = new DescriptiveStatistics();
        int projectionCounter = 0;
        for (int i = 0; i < k; i++) {
            TIntArrayList DMDi = new TIntArrayList(valD.clone());
            TIntArrayList UMUi = new TIntArrayList(valU.clone());
            DMDi.removeAll(splitsD[i]);
            UMUi.removeAll(splitsU[i]);

            IDataStore Dj = D.getView(DMDi.toArray());
            IDataStore Uj = Ui.getView(UMUi.toArray());

            IKernel trainTrainKernel = new OnTheFlyKernel(Dj, kernelFunction);
            IKernel trainTestKernel = new OnTheFlyKernel(Dj, Uj, kernelFunction);
            IKernel testTestKernel = new OnTheFlyKernel(Uj, kernelFunction);

            double constant = 0;
            for (Entry<Entry<IInstance, IInstance>, Double> entry : testTestKernel) {
                constant += entry.getValue();
            }
            constant /= (Uj.size() * Uj.size());

            MMDEstimator mmdEstimator = new MMDEstimator(trainTrainKernel, trainTestKernel);
            double[] estimate = mmdEstimator.estimateFractions();
            double objective = mmdEstimator.getObjective();
            objective += constant;

            double phipphip = 0;
            double phipphin = 0;
            int countpp = 0, countpn = 0;
            for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTrainKernel) {
                int y1 = (int) entry.getKey().getKey().label();
                int y2 = (int) entry.getKey().getValue().label();
                if (y1 == 1 && y2 == 1) {
                    phipphip += entry.getValue();
                    countpp++;
                } else if (y1 == 1 && y2 == 0) {
                    phipphin += entry.getValue();
                    countpn++;
                }
            }
            phipphip /= countpp;
            phipphin /= countpn;

            double phipphiu = 0;
            double phinphiu = 0;
            int countpu = 0, countnu = 0;
            for (Entry<Entry<IInstance, IInstance>, Double> entry : trainTestKernel) {
                int y = (int) entry.getKey().getKey().label();
                if (y == 1) {
                    phipphiu += entry.getValue();
                    countpu++;
                } else {
                    phinphiu += entry.getValue();
                    countnu++;
                }
            }
            phipphiu /= countpu;
            phinphiu /= countnu;

            double numerator = phipphip - phipphin - phipphiu + phinphiu;

            objectiveValues.addValue(objective);
            estimates.addValue(estimate[1]);
            numerators.addValue(numerator);
            if (numerator < 0 || numerator > 1)
                projectionCounter++;
        }
        IKernel trainTrainKernel = new OnTheFlyKernel(D, kernelFunction);
        IKernel trainTestKernel = new OnTheFlyKernel(D, Ui, kernelFunction);
        IKernel testTestKernel = new OnTheFlyKernel(Ui, kernelFunction);

        double constant = 0;
        for (Entry<Entry<IInstance, IInstance>, Double> entry : testTestKernel) {
            constant += entry.getValue();
        }
        constant /= (Ui.size() * Ui.size());

        MMDEstimator mmdEstimator = new MMDEstimator(trainTrainKernel, trainTestKernel);
        double[] estimate = mmdEstimator.estimateFractions();
        double objective = mmdEstimator.getObjective();
        objective += constant;

        features[0] = (float) estimate[1];
        features[1] = (float) solve(estimate[1], estimates.getVariance());
        if (features[0] <= 0)
            throw new IllegalStateException("Should have at least two real root!");
        features[2] = (float) objective;
        features[3] = Ui.size();
        features[4] = (float) objectiveValues.getStandardDeviation();
        double mu = estimates.getMean();
        features[5] = (mu > 0.9 || mu < 0.1) ? 1 : 0;
        features[6] = (float) numerators.getStandardDeviation();
        features[7] = projectionCounter;
        return features;
    }

    private double solve(double thetaMMD, double variance) {
        double v = variance;
        double md = thetaMMD;
        double a = v;
        double b = v - md + md * md;
        double c = v + 2 * md - 1 - 4 * md * md + 2 * md;
        double d = 4 * md * md - 4 * md + 1;
        double[] coeffs = { d, c, b, a };
        LaguerreSolver laguerreSolver = new LaguerreSolver();
        Complex[] solutions = laguerreSolver.solveAllComplex(coeffs, 0);
        double max = -1;
        for (int i = 0; i < solutions.length; i++) {
            double imag = solutions[i].getImaginary();
            if (imag == 0) {
                double real = solutions[i].getReal();
                if (max < real) {
                    max = real;
                }
            }
        }
        return max;
    }
    
}
