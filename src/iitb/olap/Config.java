package iitb.olap;

import gnu.trove.list.array.TIntArrayList;
import iitb.data.FeatureStoreWrapper;
import iitb.data.IDataStore;
import iitb.data.IInstance;
import iitb.shared.Utils;
import iitb.shared.XMLConfigs;
import iitb.shared.repository.fixedLengthRecords.FeatureStore;
import iitb.util.Twiddle;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

public class Config {

    private Element params;
    private String name;
    private String kernel;
    private String trainPath;
    private String trainSize;
    private String testPath;
    private String testSize;
    private Map<Integer, double[]> partitionMap;
    private Set<Integer> groupingAttributes;

    public Config(String configFile, String datasetName, String kernel, String partitionMapString) throws FileNotFoundException, ParserConfigurationException, SAXException, IOException {
        System.out.print("Loading config file ... ");
        params = XMLConfigs.load(new FileReader(configFile));
        System.out.println("[DONE]");
        this.name = datasetName;
        this.kernel = kernel;
        this.partitionMap = getValueMap(partitionMapString);
        this.groupingAttributes = this.partitionMap.keySet();

        params = XMLConfigs.getElement(params, datasetName);
        System.out.println("Running in traintest mode ...");
        String elName = "FeatureStore";
        Element featureStore = XMLConfigs.getElement(params, elName);
        System.out.print("Loading data ... ");
        featureStore = XMLConfigs.getElement(featureStore, elName);
        System.out.println("[DONE]");
        trainPath = featureStore.getAttribute("train");
        trainSize = featureStore.getAttribute("train-size");
        testPath = featureStore.getAttribute("test");
        testSize = featureStore.getAttribute("test-size");
    }

    public String getName() {
        return this.name;
    }

    public String getKernel() {
        return this.kernel;
    }

    public Set<Integer> getGroupingAttributes() {
        return this.groupingAttributes;
    }

    public String getTrainFile() {
        return this.trainPath;
    }

    public String getTestFile() {
        return this.testPath;
    }

    private static Map<Integer, double[]> getValueMap(String valueMapString) {
        valueMapString = valueMapString.trim();
        String[] attributes = valueMapString.split("\\s*;\\s*");
        TreeMap<Integer, double[]> treeMap = new TreeMap<Integer, double[]>();
        for (String attribute : attributes) {
            String[] keyRange = attribute.split("\\s*:\\s*");
            Integer attrib = Integer.parseInt(keyRange[0].trim());
            String rangeStr = keyRange[1];
            rangeStr = rangeStr.trim();
            rangeStr = rangeStr.substring(1, rangeStr.length() - 1);
            rangeStr = rangeStr.trim();
            double[] range = null;
            if (!rangeStr.isEmpty()) {
                String[] rangeVals = rangeStr.split("\\s*,\\s*");
                range = new double[rangeVals.length];
                for (int i = 0; i < rangeVals.length; i++) {
                    String val = rangeVals[i];
                    val = val.trim();
                    if (val.equalsIgnoreCase("-Inf"))
                        range[i] = Double.NEGATIVE_INFINITY;
                    else if (val.equalsIgnoreCase("+Inf") || val.equalsIgnoreCase("Inf"))
                        range[i] = Double.POSITIVE_INFINITY;
                    else
                        range[i] = Double.parseDouble(val);
                }
                Arrays.sort(range);
            }
            treeMap.put(attrib, range);
        }
        return treeMap;
    }

    private FeatureStore getTrainingData() throws SecurityException, IllegalArgumentException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, InvocationTargetException {
        String elName = "FeatureStore";
        Element featureStore = XMLConfigs.getElement(params, elName);
        featureStore = XMLConfigs.getElement(featureStore, elName);
        featureStore.setAttribute("path", trainPath);
        featureStore.setAttribute("num-instances", trainSize);
        return (FeatureStore) Utils.makeClass(params, elName);
    }

    private FeatureStore getTestingData() throws SecurityException, IllegalArgumentException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, InvocationTargetException {
        String elName = "FeatureStore";
        Element featureStore = XMLConfigs.getElement(params, elName);
        featureStore = XMLConfigs.getElement(featureStore, elName);
        featureStore.setAttribute("path", testPath);
        featureStore.setAttribute("num-instances", testSize);
        return (FeatureStore) Utils.makeClass(params, elName);
    }

    public DataHandler getDataHandler() throws Exception {
        return new DataHandler(this, partitionMap);
    }

    public static class DataHandler {

        private IDataStore trainData;
        private IDataStore testData;
        private GroupBy groupBy;
        private Map<String, Integer> baseGroupIdToIndexMap;
        private Map<Integer, String> baseGroupIndexToIdMap;
        private TIntArrayList[] basePartitions;
        private int[] baseNu;
        private int[] trainClassCounts;
        private double[] trainClassProbs;
        private int[][] trueTestClassCounts;
        private double meanDistance;

        public DataHandler(Config config, Map<Integer, double[]> partitionMap) throws Exception {
            this.trainData = new FeatureStoreWrapper(config.getTrainingData());
            this.testData = new FeatureStoreWrapper(config.getTestingData());
            System.out.print("Doing group by ... ");
            this.groupBy = new GroupBy(testData, partitionMap);
            System.out.println("[DONE]");
            SortedMap<String, TIntArrayList> baseGroups = this.groupBy.getBaseGroups();
            this.basePartitions = new TIntArrayList[baseGroups.keySet().size()];
            this.baseGroupIdToIndexMap = new HashMap<String, Integer>();
            this.baseGroupIndexToIdMap = new HashMap<Integer, String>();
            int i = 0;
            for (String key : baseGroups.keySet()) {
                this.baseGroupIdToIndexMap.put(key, i);
                this.baseGroupIndexToIdMap.put(i, key);
                this.basePartitions[i++] = baseGroups.get(key);
            }
            System.out.print("Computing group sizes ... ");
            computeGroupSizes();
            System.out.println("[DONE]");
            System.out.println("Computing train class probabilities ... ");
            computeTrainClassProbabilities();
            System.out.println("[DONE]");
            System.out.print("Compute true test class probabilities ... ");
            computeTrueTestClassProbs();
            System.out.println("[DONE]");
            System.out.print("Getting mean distance ... ");
            this.meanDistance = this.meanDistance();
            System.out.println("[DONE]");
        }

        public double getMeanDistance() {
            return this.meanDistance;
        }

        public Iterator<IInstance> getTrainingDataIterator() throws Exception {
            return trainData.iterator();
        }

        public Iterator<IInstance> getTestingDataIterator() throws Exception {
            return testData.iterator();
        }

        public int getTrainingSize() {
            return this.trainData.size();
        }

        public int getTestSize() {
            return this.testData.size();
        }

        public int getBasePartitionCount() {
            return basePartitions.length;
        }

        public TIntArrayList getBasePartitionIndices(int i) {
            return basePartitions[i];
        }

        public TIntArrayList getBasePartitionIndices(int[] indices) {
            TIntArrayList total = new TIntArrayList();
            for (int i : indices) {
                total.add(getBasePartitionIndices(i).toArray());
            }
            total.sort();
            return total;
        }

        public Iterator<IInstance> getBasePartitionIterator(int i) throws Exception {
            return testData.getView(basePartitions[i].toArray()).iterator();
        }

        public Iterator<IInstance> getBasePartitionIterator(int[] indices) throws Exception {
            TIntArrayList total = getBasePartitionIndices(indices);
            return testData.getView(total.toArray()).iterator();
        }

        private void computeGroupSizes() throws Exception {
            this.baseNu = new int[this.getBasePartitionCount()];
            for (int i = 0; i < baseNu.length; i++) {
                this.baseNu[i] = this.getBasePartitionIndices(i).size();
            }
        }
        
        public int numClasses() {
            return this.trainData.numClasses();
        }
        
        public int numFeatures() {
            return this.trainData.numFeatures();
        }

        private double meanDistance() throws Exception {
            DescriptiveStatistics stats = new DescriptiveStatistics();
            for (IInstance instance1 : trainData) {
                for (IInstance instance2 : trainData) {
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

        private void computeTrainClassProbabilities() throws Exception {
            int numY = trainData.numClasses();

            trainClassCounts = new int[numY];
            int trainClassSum = 0;
            for (IInstance instance : trainData) {
                int y = (int) instance.label();
                trainClassCounts[y]++;
                trainClassSum++;
            }
            System.out.println("Train counts=" + Arrays.toString(trainClassCounts));
            System.out.println("numf = " + trainData.numFeatures());
            trainClassProbs = new double[numY];
            for (int i = 0; i < trainClassProbs.length; i++) {
                trainClassProbs[i] = (double) trainClassCounts[i] / (double) trainClassSum;
            }
        }

        private void computeTrueTestClassProbs() throws Exception {
            int numY = trainData.numClasses();
            trueTestClassCounts = new int[this.getBasePartitionCount()][numY];
            for (int i = 0; i < this.getBasePartitionCount(); i++) {
                Iterator<IInstance> testingSet = this.getBasePartitionIterator(i);
                while (testingSet.hasNext()) {
                    IInstance instance = testingSet.next();
                    int y = (int) instance.label();
                    trueTestClassCounts[i][y]++;
                }
            }
        }

        public int[] getTrainClassCounts() {
            return this.trainClassCounts;
        }

        public double[] getTrainClassProbs() {
            return this.trainClassProbs;
        }

        public int[] getBasePartitionTrueCounts(int i) {
            return trueTestClassCounts[i];
        }

        public double[] getGroupClassProbs(int[] indices) {
            double nu = this.getGroupSize(indices);
            double[] classCounts = new double[trainClassCounts.length];
            for (int i : indices) {
                int[] baseClassCounts = getBasePartitionTrueCounts(i);
                for (int j = 0; j < baseClassCounts.length; j++) {
                    classCounts[j] += baseClassCounts[j];
                }
            }
            for (int i = 0; i < classCounts.length; i++) {
                classCounts[i] = classCounts[i] / nu;
            }
            return classCounts;
        }

        public int getBasePartitionSize(int i) {
            return this.baseNu[i];
        }

        public int getGroupSize(int[] indices) {
            int total = 0;
            for (int i : indices)
                total += this.baseNu[i];
            return total;
        }

        public int getNumAttributesGrouped() {
            return this.groupBy.getAttributes().size();
        }

        public List<int[]> getIteratorForLevel(int level) {
            level = level - 1;
            List<Integer> attributes = this.groupBy.getAttributes();
            int numAttributes = attributes.size();
            List<int[]> baseIndicators = new LinkedList<int[]>();
            if (level == 0) {
                for (int i = this.basePartitions.length - 1; i >= 0; i--) {
                    int[] indicator = new int[this.basePartitions.length];
                    indicator[i] = 1;
                    baseIndicators.add(indicator);
                }
            } else if (level == numAttributes) {
                int[] indicator = new int[this.basePartitions.length];
                for (int i = 0; i < indicator.length; i++)
                    indicator[i] = 1;
                baseIndicators.add(indicator);
            } else {
                int N = numAttributes;
                int M = numAttributes - level;
                Twiddle twiddle = new Twiddle(N, M);

                for (int[] attIndicator : twiddle) {
                    for (GroupInfo groupInfo : this.getBaseIndicators(attIndicator, attributes, M))
                        baseIndicators.add(groupInfo.baseIndicators);
                }
            }
            return baseIndicators;
        }
        
        public class GroupInfo {
            int[] baseIndicators;
            int[] attIndicators;
            List<Number> values;
            
            public boolean isChild(GroupInfo child) {
                boolean attFlag = true;
                for (int i = 0; i < child.attIndicators.length; i++) {
                    if (attIndicators[i] == 1 && child.attIndicators[i] == 0) {
                        attFlag = false;
                    }
                }
                
                if (!attFlag)
                    return false;
                
                boolean valueFlag = true;
                for (int i = 0, j = 0, k = 0; i < child.attIndicators.length; i++) {
                    if (attIndicators[i] == 1) {
                        Number parentVal = values.get(j++);
                        Number childVal = child.values.get(k);
                        if (!parentVal.equals(childVal))
                            valueFlag = false;
                    }
                    if (child.attIndicators[i] == 1)
                        k++;
                }
                
                if (!valueFlag)
                    return false;
                return true;
            }
        }
        
        public List<GroupInfo> getWrappedIteratorForLevel(int level) {
            level = level - 1;
            List<Integer> attributes = this.groupBy.getAttributes();
            int groupedAttributes = attributes.size();
            List<GroupInfo> baseIndicators = new LinkedList<GroupInfo>();
            if (level == 0) {
                int[] attIndicator = new int[groupedAttributes];
                for (int i = 0; i < attIndicator.length; i++) {
                    attIndicator[i] = 1;
                }
                for (int i = this.basePartitions.length - 1; i >= 0; i--) {
                    int[] indicator = new int[this.basePartitions.length];
                    indicator[i] = 1;
                    GroupInfo groupInfo = new GroupInfo();
                    groupInfo.attIndicators = attIndicator.clone();
                    groupInfo.baseIndicators = indicator.clone();
                    groupInfo.values = new LinkedList<Number>(); 
                    for (Entry <Integer, Number> attValue : this.groupBy.getAttrValForBaseGroup(this.baseGroupIndexToIdMap.get(i)).entrySet()) {
                        groupInfo.values.add(attValue.getValue());
                    }
                    baseIndicators.add(groupInfo);
                }
            } else if (level == groupedAttributes) {
                int[] indicator = new int[this.basePartitions.length];
                for (int i = 0; i < indicator.length; i++)
                    indicator[i] = 1;
                int[] attIndicator = new int[groupedAttributes];
                GroupInfo groupInfo = new GroupInfo();
                groupInfo.baseIndicators = indicator;
                groupInfo.attIndicators = attIndicator;
                groupInfo.values = new LinkedList<Number>(); 
                baseIndicators.add(groupInfo);
            } else {
                int N = groupedAttributes;
                int M = groupedAttributes - level;
                Twiddle twiddle = new Twiddle(N, M);

                for (int[] attIndicator : twiddle) {
                    baseIndicators.addAll(this.getBaseIndicators(attIndicator, attributes, M));
                }
            }
            return baseIndicators;
        }

        private List<GroupInfo> getBaseIndicators(int[] attIndicator, List<Integer> attributes, int M) {
            List<GroupInfo> baseIndicators = new LinkedList<GroupInfo>();
            List<SortedSet<Number>> activeIDList = new LinkedList<SortedSet<Number>>();
            List<Integer> activeAttributes = new LinkedList<Integer>();
            for (int i = 0; i < attIndicator.length; i++) {
                if (attIndicator[i] == 1) {
                    SortedSet<Number> activeIDs = this.groupBy.getActiveIDs(attributes.get(i));
                    activeIDList.add(activeIDs);
                    activeAttributes.add(attributes.get(i));
                }
            }

            Stack<Iterator<Number>> iteratorStack = new Stack<Iterator<Number>>();
            Stack<Number> valueStack = new Stack<Number>();
            for (SortedSet<Number> activeIDs : activeIDList) {
                Iterator<Number> iterator = activeIDs.iterator();
                if (iterator.hasNext())
                    valueStack.add(iterator.next());
                iteratorStack.push(iterator);
            }

            boolean done = false;
            while (!done) {
                SortedSet<String> result = null;
                for (int i = 0; i < valueStack.size(); i++) {
                    Integer attribute = activeAttributes.get(i);
                    Number value = valueStack.get(i);
                    SortedSet<String> baseGroups = this.groupBy.getBaseGroups(attribute, value);
                    if (result == null) {
                        if (baseGroups != null)
                            result = new TreeSet<String>(baseGroups);
                        else
                            result = new TreeSet<String>();
                    } else {
                        if (baseGroups != null)
                            result.retainAll(baseGroups);
                    }
                }
                if (result != null && !result.isEmpty()) {
                    int[] baseIndicator = new int[this.getBasePartitionCount()];
                    for (String groupID : result) {
                        baseIndicator[this.baseGroupIdToIndexMap.get(groupID)] = 1;
                    }
                    
                    GroupInfo groupInfo = new GroupInfo();
                    groupInfo.baseIndicators = baseIndicator.clone();
                    groupInfo.attIndicators = attIndicator.clone();
                    groupInfo.values = new LinkedList<Number>(valueStack);
                    baseIndicators.add(groupInfo);
                }

                Iterator<Number> top = iteratorStack.pop();
                while (iteratorStack.size() < activeAttributes.size()) {
                    if (top.hasNext()) {
                        valueStack.pop();
                        valueStack.push(top.next());
                        iteratorStack.push(top);
                        for (int i = iteratorStack.size(); i < activeAttributes.size(); i++) {
                            Iterator<Number> iterator = activeIDList.get(i).iterator();
                            iteratorStack.push(iterator);
                            if (iterator.hasNext())
                                valueStack.push(iterator.next());
                            else
                                throw new IllegalStateException();
                        }
                    } else {
                        if (iteratorStack.isEmpty()) {
                            done = true;
                            break;
                        }
                        top = iteratorStack.pop();
                        valueStack.pop();
                    }
                }
            }
            
            return baseIndicators;
        }

        public static void main(String[] args) throws Exception {
            String configFile = "configcons.xml";

            StringBuilder builder = new StringBuilder();
            builder.append("-Inf");
            for (double x = -0.9999; x < 1; x += 0.0001) {
                builder.append(", " + x);
            }
            builder.append(", Inf");

            Config config = new Config(configFile, "GroupTest", "Gaussian", "0:[" + builder.toString() + "]; 1:[" + builder.toString() + "]; 2:[" + builder.toString() + "]");
            DataHandler dataHandler = config.getDataHandler();
            System.out.println(dataHandler.groupBy.getActiveIDs(0));
            System.out.println(dataHandler.groupBy.getActiveIDs(1));
            System.out.println(dataHandler.getBasePartitionCount());
            System.out.println(dataHandler.groupBy.getBaseGroups().keySet());
            for (int[] indicators : dataHandler.getIteratorForLevel(3)) {
                TIntArrayList indices = new TIntArrayList();
                for (int i = 0; i < indicators.length; i++) {
                    if (indicators[i] == 1)
                        indices.add(i);
                }
                HashSet<String> keySet = new HashSet<String>();
                Iterator<IInstance> iterator = dataHandler.getBasePartitionIterator(indices.toArray());
                while (iterator.hasNext()) {
                    IInstance instance = iterator.next();
                    StringBuilder stringBuilder = new StringBuilder();
                    stringBuilder.append(instance.get(0)).append('_');
                    stringBuilder.append(instance.get(1)).append('_');
                    stringBuilder.append(instance.get(2));
                    keySet.add(stringBuilder.toString());
                }
                System.out.println(keySet);
            }
        }

    }

}