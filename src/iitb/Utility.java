package iitb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Properties;
import java.util.Random;

import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.data.WekaStoreWrapper;
import iitb.kernel.IKernelFunction;
import iitb.kernel.WeightedKernelFunction;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Utility {

    public static double meanDistance(IDataStore dataStore) throws Exception {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (IInstance instance1 : dataStore) {
            for (IInstance instance2 : dataStore) {
                double distance = 0;
                for (int i = 0; i < instance1.numFeatures(); i++) {
                    double val = instance1.get(i) - instance2.get(i);
                    distance = distance + val * val;
                }
                stats.addValue(Math.sqrt(distance));
            }
        }
        return stats.getMean();
    }
    
    public static double[] getClassProps(IDataStore dataStore) {
        double[] props = new double[dataStore.numClasses()];
        for (IInstance instance : dataStore) {
            int label = (int) instance.label();
            props[label]++;
        }
        for (int i = 0; i < props.length; i++)
            props[i] /= dataStore.size();
        return props;
    }
    
    public static TIntArrayList[] splitter(IDataStore source, int k, long seed) {
        Random random = new Random(seed);
        int size = source.size();
        TIntArrayList[] splits = new TIntArrayList[k];
        for (int i = 0; i < k; i++)
            splits[i] = new TIntArrayList();
        for (int i = 0; i < size; i++) {
            int j = random.nextInt(k);
            splits[j].add(i);
        }
        return splits;
    }
    
    public static WeightedKernelFunction<IKernelFunction> readKernelFile(String file) throws Exception {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
        String line = null;
        WeightedKernelFunction<IKernelFunction> weightedKernelFunction = new WeightedKernelFunction<IKernelFunction>();
        while ((line = bufferedReader.readLine()) != null) {
            String[] cols = line.split("#");
            String clazzName = cols[0];
            String parameters = cols[1];
            String weight = cols[2];
            IKernelFunction kernelFunction = (IKernelFunction) Class.forName(clazzName).newInstance();
            Properties properties = new Properties();
            for (String parameter : parameters.split(";")) {
                String[] keyVal = parameter.split("=");
                properties.setProperty(keyVal[0], keyVal[1]);
            }
            kernelFunction.setParameters(properties);
            weightedKernelFunction.addKernelFunction(kernelFunction, Float.valueOf(weight));
        }
        bufferedReader.close();
        return weightedKernelFunction;
    }

    public static IDataStore[] getPrototypes(IDataStore source, long seed, int numProportions, int numPrototypesPerProportion, int sizePerPrototype) {
        Random random = new Random(seed);
        int numClasses = source.numClasses();
        GammaDistribution[] gammaDistributions = new GammaDistribution[numClasses];
        for (int i = 0; i < numClasses; i++) {
            gammaDistributions[i] = new GammaDistribution(1, 1);
            gammaDistributions[i].reseedRandomGenerator(seed);
        }
        float[][] simplexSamples = getSimplexSamples(numClasses, numProportions, gammaDistributions);
        IDataStore[] prototypes = new IDataStore[numProportions*numPrototypesPerProportion];
        for (int i = 0, k = 0; i < numProportions; i++) {
            for (int j = 0; j < numPrototypesPerProportion; j++, k++) {
                prototypes[k] = getSample(source, simplexSamples[i], sizePerPrototype, random);
            }
        }
        return prototypes;
    }

    private static int[] getCount(float[] proportions, int size) {
        int[] counts = new int[proportions.length];
        int sum = 0, max = 0, maxIndex = -1;
        for (int i = 0; i < counts.length; i++) {
            counts[i] = Math.round(proportions[i] * size);
            sum += counts[i];
            if (max < counts[i]) {
                max = counts[i];
                maxIndex = i;
            }
        }
        if (sum < size) {
            counts[maxIndex] += (size - sum);
        } else if (sum > size) {
            counts[maxIndex] -= (sum - size);
        }
        return counts;
    }

    public static float[][] getSimplexSamples(int corners, int numSamples, GammaDistribution[] gammaDistributions) {
        if (corners == 2 && numSamples == 11 && allEqualShapes(gammaDistributions)) {
            return new float[][] { { 0.01f, 0.99f }, { 0.1f, 0.9f }, { 0.2f, 0.8f }, { 0.3f, 0.7f }, { 0.4f, 0.6f }, { 0.5f, 0.5f }, { 0.6f, 0.4f }, { 0.7f, 0.3f }, { 0.8f, 0.2f }, { 0.1f, 0.9f }, { 0.99f, 0.01f } };
        } else {
            float[][] simplexSamples = new float[numSamples][corners];
            for (int i = 0; i < numSamples; i++) {
                float sum = 0;
                for (int j = 0; j < gammaDistributions.length; j++) {
                    simplexSamples[i][j] = (float) gammaDistributions[j].sample();
                    sum += simplexSamples[i][j];
                }
                for (int j = 0; j < gammaDistributions.length; j++) {
                    simplexSamples[i][j] /= sum;
                }
            }
            return simplexSamples;
        }
    }

    private static boolean allEqualShapes(GammaDistribution[] gammaDistributions) {
        for (int i = 0; i < gammaDistributions.length - 1; i++) {
            double shape_i = gammaDistributions[i].getShape();
            double shape_ip1 = gammaDistributions[i + 1].getShape();
            if (shape_i != shape_ip1)
                return false;
        }
        return true;
    }

    public static IDataStore getSample(IDataStore dataStore, float[] proportions, int size, Random random) {
        int[] counts = getCount(proportions, size);
        TIntObjectHashMap<TIntArrayList> classToInstanceList = new TIntObjectHashMap<TIntArrayList>();
        for (int i = 0; i < counts.length; i++) {
            classToInstanceList.put(i, new TIntArrayList());
        }

        int[] currentIndex = new int[counts.length];
        int index = 0;
        for (IInstance instance : dataStore) {
            int y = (int) instance.label();
            if (currentIndex[y] < counts[y]) {
                classToInstanceList.get(y).add(index);
            } else {
                if (currentIndex[y] != 0) {
                    int j = random.nextInt(currentIndex[y]);
                    if (j < counts[y]) {
                        classToInstanceList.get(y).set(j, index);
                    }
                }
            }
            currentIndex[y]++;
            index++;
        }

        TIntArrayList indexList = new TIntArrayList();
        for (int i = 0; i < classToInstanceList.size(); i++) {
            TIntArrayList list = classToInstanceList.get(i);
            for (int j = 0; j < list.size(); j++) {
                indexList.add(list.get(j));
            }
        }
        indexList.sort();

        return dataStore.getView(indexList.toArray());
    }

    public static double selectBandwidth(IDataStore dataStore) throws Exception {
        int numFeatures = dataStore.numFeatures();
        int numTraining = dataStore.size();
        double meanStdX = 0;
        if (numFeatures > 1e4) {
            meanStdX = mean(std(dataStore, 0, 10, numFeatures));
        } else {
            meanStdX = mean(std(dataStore, 0, 1, numFeatures));
        }
        double sigma = 1;
        if (numTraining < 500)
            sigma = 0.4 * Math.sqrt(numFeatures);
        else if (numTraining < 1000)
            sigma = 0.2 * Math.sqrt(numFeatures);
        else
            sigma = 0.14 * Math.sqrt(numFeatures);
        return sigma * meanStdX;
    }

    public static double[] std(IDataStore dataStore, int start, int step, int end) throws Exception {
        int numVals = 0;
        for (int i = start; i < end; i += step) {
            numVals++;
        }

        DescriptiveStatistics stats[] = new DescriptiveStatistics[numVals];
        for (int i = 0; i < stats.length; i++)
            stats[i] = new DescriptiveStatistics();

        for (IInstance instance : dataStore) {
            for (int i = start, j = 0; i < end; i += step, j++) {
                double x = instance.get(i);
                stats[j].addValue(x);
            }
        }

        double[] stds = new double[numVals];
        for (int i = 0; i < stats.length; i++) {
            stds[i] = stats[i].getStandardDeviation();
        }
        return stds;
    }

    public static double mean(double[] x) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (double xi : x) {
            stats.addValue(xi);
        }
        return stats.getMean();
    }
    
    public static IDataStore readCSVFile(String file, String separator, boolean header) throws Exception {
        return readCSVFile(file, separator, header, false);
    }

    public static IDataStore readCSVFile(String file, String separator, boolean header, boolean isLabelNumeric) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(separator);
        loader.setFile(new File(file));
        loader.setNoHeaderRowPresent(!header);
        Instances dataset = loader.getDataSet();
        if (!isLabelNumeric) {
            NumericToNominal numericToNominal = new NumericToNominal();
            numericToNominal.setAttributeIndices("last");
            numericToNominal.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, numericToNominal);
        }
        dataset.setClassIndex(dataset.numAttributes() - 1);
        return new WekaStoreWrapper(dataset);
    }
    
    public static IDataStore readLSVMFile(String file) throws Exception {
        return readLSVMFile(file, false);
    }

    public static IDataStore readLSVMFile(String file, boolean isLabelNumeric) throws Exception {
        LibSVMLoader loader = new LibSVMLoader();
        loader.setFile(new File(file));
        Instances dataset = loader.getDataSet();
        return new WekaStoreWrapper(dataset);
    }
    
    public static TIntArrayList selectKRandom(Random rand, int n, int k) {
        int array[] = new int[n];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }
        int p = Math.min(k, array.length);
        TIntArrayList selElems = new TIntArrayList();
        for (int i = 0; i < p; i++) {
            int rem = array.length-i;
            int swapPos = i + rand.nextInt(rem);
            int oldVal = array[swapPos];
            array[swapPos] = array[i];
            array[i] = oldVal;
            selElems.add(array[i]);
        }
        selElems.sort();
        return selElems;
    }

}
