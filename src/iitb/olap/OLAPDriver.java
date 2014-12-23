package iitb.olap;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import iitb.Utility;
import iitb.cre.MMDEstimator;
import iitb.data.IInstance;
import iitb.kernel.GaussianFunction;
import iitb.kernel.IKernelFunction;
import iitb.olap.Config.DataHandler;
import iitb.olap.Config.DataHandler.GroupInfo;
import iitb.olap.MMDConsistency.WeightScheme;

public class OLAPDriver {

    public static void main(String[] args) throws Exception {
        String dataset = "SatImage";
        String outputPrefix = dataset + "2_cons_output_";
        int numFeatures = 36;
        Random random = new Random(1);
        for (int i = 0; i < 10; i++) {
            TIntArrayList perm = Utility.selectKRandom(random, numFeatures, 3);
//            TIntArrayList perm = new TIntArrayList();
//            perm.add(4); perm.add(6); perm.add(7);
            main2(dataset, perm.toArray(), outputPrefix + Arrays.toString(perm.toArray()) + ".csv");
        }
    }
    
    public static void main2(String dataset, int[] atts, String outputFile) throws Exception {
        // String configFile = args[0];
        String configFile = "configcons.xml";

        StringBuilder builder = new StringBuilder();
        builder.append("-Inf");
        for (double x = -0.9; x < 1; x += 0.1) {
            builder.append(", " + x);
        }
        builder.append(", Inf");

        String attributeStr = atts[0] + ":[" + builder.toString() + "]";
        for (int i = 1; i < atts.length; i++) {
            attributeStr += "; " + atts[i] + ":[" + builder.toString() + "]";
        }
        
        Config config = new Config(configFile, dataset, "Gaussian", attributeStr);

        System.out.println("Loading data handler ... ");
        DataHandler dataHandler = config.getDataHandler();
        double sigma = dataHandler.getMeanDistance();
        IKernelFunction kernel = new GaussianFunction(sigma);
        System.out.print("Building A and b ... ");
        long start = System.currentTimeMillis();
        double[][] A = getA(dataHandler, kernel, config);
        double[][] b = getb(dataHandler, kernel, config);
        long end = System.currentTimeMillis();
        System.out.println("[DONE] (" + ((end - start) / 1000) + " ms)");
        
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile));
        bufferedWriter.write("level\t groups\t size\t method\t ");
        for (int i = 1; i <= A.length; i++) {
            bufferedWriter.write("tr" + i + "\t ");
        }
        for (int i = 1; i <= A.length; i++) {
            bufferedWriter.write("pr" + i + "\t ");
        }
        bufferedWriter.write("acc");
        bufferedWriter.newLine();
        bufferedWriter.flush();

//        doCheckConsistency(A, b, dataHandler);
        doMMDCR(A, b, dataHandler, bufferedWriter);
        doMMDCRStraw(A, b, dataHandler, bufferedWriter);
        doMMDConsistency(A, b, dataHandler, WeightScheme.ALLEQUAL, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.ROOTSCHEME, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.PROPWEIGHTSCHEME, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.INVPROPWEIGHTSCHEME, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.PROPCHILDSCHEME, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.PROPCHILDWEIGHTSCHEME, bufferedWriter);
//        doMMDConsistency(A, b, dataHandler, WeightScheme.PROPCHILDINVWEIGHTSCHEME, bufferedWriter);
        doMMDConsistency(A, b, dataHandler, WeightScheme.DISPARITYSCHEME, bufferedWriter);
        bufferedWriter.close();
    }

    private static void doMMDCR(double[][] A, double[][] b, DataHandler dataHandler, BufferedWriter bufferedWriter) throws Exception {
        for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                doWrite(dataHandler, baseIndicator, A, b, level, bufferedWriter, "PlainMMD");
            }
        }
    }
    
    private static void doCheckConsistency(double[][] A, double[][] b, DataHandler dataHandler) throws Exception {
        TDoubleArrayList inconsistencies = new TDoubleArrayList();
        for (int level = 2; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            System.out.print("Comparing level " + level + " with level " + (level - 1));
            double levelError = 0;
            for (GroupInfo parentInfo : dataHandler.getWrappedIteratorForLevel(level)) {
                double[] parentEstimate = getMMDCREstimate(dataHandler, parentInfo.baseIndicators, A, b, level);
                double parentSize = getSize(dataHandler, parentInfo.baseIndicators);
                
                LinkedList<double[]> childEstimates = new LinkedList<double[]>();
                LinkedList<Double> childSizes = new LinkedList<Double>();
                LinkedList<GroupInfo> children = new LinkedList<GroupInfo>();
                for (GroupInfo childInfo : dataHandler.getWrappedIteratorForLevel(level - 1)) {
                    if (parentInfo.isChild(childInfo)) {
                        children.add(childInfo);
                    }
                }
                
                removeConflictingChildren(children, parentInfo);
                
                for (GroupInfo childInfo : children) {
                    double[] estimate = getMMDCREstimate(dataHandler, childInfo.baseIndicators, A, b, level);
                    childEstimates.add(estimate);
                    childSizes.add(getSize(dataHandler, childInfo.baseIndicators));
                }
                
                double[] consistentParent = new double[parentEstimate.length];
                for (int i = 0; i < childEstimates.size(); i++) {
                    double[] childEstimate = childEstimates.get(i);
                    double childSize = childSizes.get(i);
                    for (int j = 0; j < consistentParent.length; j++) {
                        consistentParent[j] += (childSize / parentSize) * childEstimate[j];
                    }
                }
                
                double error = 0; 
                for (int j = 0; j < parentEstimate.length; j++) {
                    error = Math.max(error, Math.abs(parentEstimate[j] - consistentParent[j])); 
                }
                levelError = Math.max(levelError, error);
                inconsistencies.add(error);
            }
            System.out.println(" " + levelError);
        }
        inconsistencies.sort();
        System.out.println(inconsistencies);
    }
    
    private static void removeConflictingChildren(List<GroupInfo> children, GroupInfo parent) {
        HashMap<String, List<GroupInfo>> duplicateBaseIndicatorMap = new HashMap<String, List<GroupInfo>>();
        HashMap<Integer, List<GroupInfo>> baseToChildMap = new HashMap<Integer, List<GroupInfo>>();
        for (GroupInfo info : children) {
            String string = Arrays.toString(info.baseIndicators);
            List<GroupInfo> list = duplicateBaseIndicatorMap.get(string);
            if (list == null) {
                list = new LinkedList<GroupInfo>();
                duplicateBaseIndicatorMap.put(string, list);
            }
            list.add(info);
            
            for (int i = 0; i < info.baseIndicators.length; i++) {
                if (info.baseIndicators[i] == 1) {
                    List<GroupInfo> list2 = baseToChildMap.get(i);
                    if (list2 == null) {
                        list2 = new LinkedList<GroupInfo>();
                        baseToChildMap.put(i, list2);
                    }
                    list2.add(info);
                }
            }
        }
        
        boolean flag = true;
        List<GroupInfo> mustRetain = new LinkedList<GroupInfo>();
        for (Entry<Integer, List<GroupInfo>> entry : baseToChildMap.entrySet()) {
            List<GroupInfo> list = entry.getValue();
            if (list.size() > 1)
                flag = false;
            else {
                mustRetain.add(list.get(0));
            }
        }
        
        if (flag)
            return;
        
        for (Entry<Integer, List<GroupInfo>> entry : baseToChildMap.entrySet()) {
            List<GroupInfo> list = entry.getValue();
            boolean canRetain = false;
            for (GroupInfo info : list) {
                if (mustRetain.contains(info))
                    canRetain = true;
            }
            if (canRetain)
                list.retainAll(mustRetain);
        }
        
        flag = true;
        for (Entry<Integer, List<GroupInfo>> entry : baseToChildMap.entrySet()) {
            List<GroupInfo> list = entry.getValue();
            if (list.size() > 1)
                flag = false;
        }
        
        if (flag) {
            children.retainAll(mustRetain);
            return;
        }
        
        for (Entry<Integer, List<GroupInfo>> entry : baseToChildMap.entrySet()) {
            List<GroupInfo> list = entry.getValue();
            boolean allNecessary = true;
            for (GroupInfo info : list) {
                allNecessary = allNecessary && (mustRetain.contains(info));
            }
            if (allNecessary)
                throw new IllegalStateException("Strange circumstance!");
            else
                mustRetain.add(list.get(0));
        }
        
        children.retainAll(mustRetain);
    }

    private static void doMMDCRStraw(double[][] A, double[][] b, DataHandler dataHandler, BufferedWriter bufferedWriter) throws Exception {
        HashMap<Integer, double[]> predMap = new HashMap<Integer, double[]>();

        for (int level = 1; level <= 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                double[] pred = doWrite(dataHandler, baseIndicator, A, b, level, bufferedWriter, "PlainMMD_Corr");
                for (int i = 0; i < baseIndicator.length; i++) {
                    if (baseIndicator[i] == 1)
                        predMap.put(i, pred);
                }
            }
        }
        for (int level = 2; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                doWrite(dataHandler, baseIndicator, predMap, level, bufferedWriter, "PlainMMD_Corr");
            }
        }
    }

    private static void doMMDConsistency(double[][] A, double[][] b, DataHandler dataHandler, WeightScheme scheme, BufferedWriter bufferedWriter) throws Exception {
        int[] nu = new int[b.length];
        for (int i = 0; i < nu.length; i++)
            nu[i] = dataHandler.getBasePartitionSize(i);
        MMDConsistency mmdc = new MMDConsistency(A, b, nu, scheme, dataHandler);
        double[] estimates = mmdc.getEstimates();
        System.out.println("scheme = " + scheme + " :: " + mmdc.getObjective());
        HashMap<Integer, double[]> predMap = new HashMap<Integer, double[]>();

        for (int i = 0; i < dataHandler.getBasePartitionCount(); i++) {
            int shift = i * A.length;
            double[] prs = new double[dataHandler.numClasses()];
            for (int j = 0; j < prs.length; j++)
                prs[j] = estimates[shift + j];
            predMap.put(i, prs);
        }

        for (int level = 1; level <= dataHandler.getNumAttributesGrouped() + 1; level++) {
            for (int[] baseIndicator : dataHandler.getIteratorForLevel(level)) {
                doWrite(dataHandler, baseIndicator, predMap, level, bufferedWriter, "JointMMD_" + scheme);
            }
        }
    }

    private static double[] doWrite(DataHandler dataHandler, int[] baseIndicator, double[][] A, double[][] b, int level, BufferedWriter bufferedWriter, String method) throws Exception {
        TIntArrayList indicesList = new TIntArrayList();
        StringBuilder builder = new StringBuilder("SEQ_");
        for (int j = 0; j < baseIndicator.length; j++) {
            if (baseIndicator[j] == 1) {
                indicesList.add(j);
                builder.append('1');
            } else {
                builder.append('0');
            }
        }
        int[] indices = indicesList.toArray();

        double nu = dataHandler.getGroupSize(indices);

        double[] newB = new double[dataHandler.numClasses()];
        for (int j : indices) {
            for (int k = 0; k < newB.length; k++) {
                newB[k] += (dataHandler.getBasePartitionSize(j) * b[j][k]) / nu;
            }
        }

        MMDEstimator mmdEstimator = new MMDEstimator(A, newB);
        bufferedWriter.write(level + "\t" + builder.toString() + "\t" + nu + "\t " + method + "\t");
        double[] trueProbs = dataHandler.getGroupClassProbs(indices);
        double[] estimate = mmdEstimator.estimateFractions();
        if (!builder.toString().contains("0"))
            System.out.println(builder.toString() + " :: " + mmdEstimator.getObjective());
        for (int j = 0; j < trueProbs.length; j++) {
            bufferedWriter.write(trueProbs[j] + "\t");
        }
        double acc = 0;
        for (int j = 0; j < trueProbs.length; j++) {
            bufferedWriter.write(estimate[j] + "\t");
            // acc = acc + Math.abs(estimate[j] - trueProbs[j]);
            acc = Math.max(acc, Math.abs(estimate[j] - trueProbs[j]));
        }
        // acc = acc / trueProbs.length;
        bufferedWriter.write("" + acc);
        bufferedWriter.newLine();
        return estimate;
    }
    
    private static double[] getMMDCREstimate(DataHandler dataHandler, int[] baseIndicator, double[][] A, double[][] b, int level) throws Exception {
        TIntArrayList indicesList = new TIntArrayList();
        StringBuilder builder = new StringBuilder("SEQ_");
        for (int j = 0; j < baseIndicator.length; j++) {
            if (baseIndicator[j] == 1) {
                indicesList.add(j);
                builder.append('1');
            } else {
                builder.append('0');
            }
        }
        int[] indices = indicesList.toArray();
        double nu = dataHandler.getGroupSize(indices);
        double[] newB = new double[dataHandler.numClasses()];
        for (int j : indices) {
            for (int k = 0; k < newB.length; k++) {
                newB[k] += (dataHandler.getBasePartitionSize(j) * b[j][k]) / nu;
            }
        }

        MMDEstimator mmdEstimator = new MMDEstimator(A, newB);
        double[] trueProbs = dataHandler.getGroupClassProbs(indices);
        double[] estimate = mmdEstimator.estimateFractions();
//        if (!builder.toString().contains("0"))
//            System.out.println(builder.toString() + " :: " + mmdEstimator.getObjective());
        double acc = 0;
        for (int j = 0; j < trueProbs.length; j++) {
            // acc = acc + Math.abs(estimate[j] - trueProbs[j]);
            acc = Math.max(acc, Math.abs(estimate[j] - trueProbs[j]));
        }
        // acc = acc / trueProbs.length;
        return estimate;
    }

    private static void doWrite(DataHandler dataHandler, int[] baseIndicator, HashMap<Integer, double[]> predMap, int level, BufferedWriter bufferedWriter, String method) throws Exception {
        TIntArrayList indicesList = new TIntArrayList();
        double[] estimate = new double[dataHandler.numClasses()];
        StringBuilder builder = new StringBuilder("SEQ_");
        for (int j = 0; j < baseIndicator.length; j++) {
            if (baseIndicator[j] == 1) {
                indicesList.add(j);
                builder.append('1');
                double nj = dataHandler.getBasePartitionSize(j);
                double[] pred = predMap.get(j);
                for (int i = 0; i < estimate.length; i++) {
                    estimate[i] = estimate[i] + nj * pred[i];
                }
            } else {
                builder.append('0');
            }
        }
        int[] indices = indicesList.toArray();
        double nu = dataHandler.getGroupSize(indices);
        for (int i = 0; i < estimate.length; i++)
            estimate[i] = estimate[i] / nu;

        bufferedWriter.write(level + "\t" + builder.toString() + "\t" + nu + "\t " + method + "\t");
        double[] trueProbs = dataHandler.getGroupClassProbs(indices);
        for (int j = 0; j < trueProbs.length; j++) {
            bufferedWriter.write(trueProbs[j] + "\t");
        }
        double acc = 0;
        for (int j = 0; j < trueProbs.length; j++) {
            bufferedWriter.write(estimate[j] + "\t");
            // acc = acc + Math.abs(estimate[j] - trueProbs[j]);
            acc = Math.max(acc, Math.abs(estimate[j] - trueProbs[j]));
        }
        // acc = acc / trueProbs.length;
        bufferedWriter.write("" + acc);
        bufferedWriter.newLine();
    }

    private static boolean debug = false;

    private static double[][] getA(DataHandler dataHandler, IKernelFunction kernel, Config config) throws Exception {
        int numTraining = dataHandler.getTrainingSize();
        int numTesting = dataHandler.getTestSize();

        IInstance[] training = new IInstance[numTraining];
        IInstance[] testing = new IInstance[numTesting];

        Iterator<IInstance> iter = dataHandler.getTrainingDataIterator();
        for (int i = 0; iter.hasNext(); i++) {
            training[i] = (IInstance) iter.next().clone();
        }

        iter = dataHandler.getTestingDataIterator();
        for (int i = 0; iter.hasNext(); i++)
            testing[i] = (IInstance) iter.next().clone();

        int numY = dataHandler.numClasses();
        double[][] A = new double[numY][numY];

        if (debug)
            return A;
        System.out.print("Constructing A ... ");
        long start = System.currentTimeMillis();
        for (int i = 0; i < numTraining; i++) {
            IInstance instance = training[i];
            int y = (int) instance.label();

            for (int j = 0; j < numTraining; j++) {
                IInstance instance1 = training[j];
                int yp = (int) instance1.label();

//                double sim = kernel.get(instance, instance1, config.getGroupingAttributes());
                double sim = kernel.get(instance, instance1);
                if (!Double.isNaN(sim)) {
                    A[y][yp] += sim;
                } else
                    throw new IllegalStateException();
            }
        }

        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A.length; j++) {
                A[i][j] = A[i][j] / (dataHandler.getTrainClassCounts()[i] * dataHandler.getTrainClassCounts()[j]);
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("[DONE] (" + (end - start) + " ms)");
        return A;
    }

    private static double[][] getb(DataHandler dataHandler, IKernelFunction kernel, Config config) throws Exception {
        int numTraining = dataHandler.getTrainingSize();
        System.out.println(dataHandler.getBasePartitionCount());
        IInstance[] training = new IInstance[numTraining];

        Iterator<IInstance> iter = dataHandler.getTrainingDataIterator();
        for (int i = 0; iter.hasNext(); i++) {
            training[i] = (IInstance) iter.next().clone();
        }

        int numY = dataHandler.numClasses();
        double[][] b = new double[dataHandler.getBasePartitionCount()][numY];
        if (debug)
            return b;
        for (int cnt = 0; cnt < dataHandler.getBasePartitionCount(); cnt++) {
            int numTesting = dataHandler.getBasePartitionSize(cnt);
            IInstance[] testing = new IInstance[numTesting];

            iter = dataHandler.getBasePartitionIterator(cnt);
            for (int i = 0; iter.hasNext(); i++) {
                testing[i] = (IInstance) iter.next().clone();
            }

            System.out.print("Constructing b ... ");
            long start = System.currentTimeMillis();
            for (int i = 0; i < numTraining; i++) {
                IInstance instance = training[i];
                int y = (int) instance.label();

                for (int j = 0; j < numTesting; j++) {
                    IInstance instance1 = testing[j];

//                    double sim = kernel.get(instance, instance1, config.getGroupingAttributes());
                    double sim = kernel.get(instance, instance1);
                    
                    if (!Double.isNaN(sim)) {
                        b[cnt][y] += sim;
                    } else
                        throw new IllegalStateException();
                }
            }

            for (int i = 0; i < b[cnt].length; i++) {
                b[cnt][i] = b[cnt][i] / (dataHandler.getTrainClassCounts()[i] * dataHandler.getBasePartitionSize(cnt));
            }

            long end = System.currentTimeMillis();
            System.out.println("[DONE] (" + (end - start) + " ms)");
        }
        return b;
    }
    
    private static double getSize(DataHandler dataHandler, int[] baseIndicator) {
        int size = 0;
        for (int i = 0; i < baseIndicator.length; i++) {
            if (baseIndicator[i] == 1)
                size += dataHandler.getBasePartitionSize(i);
        }
        return size;
    }

}
