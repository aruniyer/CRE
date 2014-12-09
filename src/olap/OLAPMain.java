package olap;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.AbstractLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.LibSVMLoader;

public class OLAPMain {
    
    public static final String CSV = "csv";
    public static final String TSV = "tsv";
    public static final String LIBSVM = "libsvm";
    
    public static Instances getInstances(String file, String type) throws Exception {
	BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(file));
	AbstractLoader loader = null;
	if (type.equalsIgnoreCase(CSV) || type.equalsIgnoreCase(TSV))
	    loader = new CSVLoader();
	else if (type.equalsIgnoreCase(LIBSVM))
	    loader = new LibSVMLoader();
	else
	    throw new RuntimeException("Unrecognized file type!");
	loader.setSource(bufferedInputStream);
	return loader.getDataSet();
    }

    // TODO: Error Handling
    public static Map<Integer, double[]> getValueMap(String valueMapString) {
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
    
    public static File createTraining(Instances instances, String outputDir, String outputFilePrefix, SortedSet<Integer> attributes, int countPerClass) throws Exception {
	HashMap<Integer, LinkedList<String>> classGroup = new HashMap<Integer, LinkedList<String>>();
	Random random = new Random(100);

	for (int i = 0; i < instances.numInstances(); i++) {
	    Instance instance = instances.instance(i);
	    LinkedList<String> lines = classGroup.get(instance.classValue());
	    if (lines == null) {
		lines = new LinkedList<String>();
		classGroup.put((int) instance.classValue(), lines);
	    }
	    StringBuilder lineBuilder = new StringBuilder();
	    for (int j = 0; j < instance.numAttributes(); j++) {
		if (!attributes.contains(j))
		    lineBuilder.append(instance.value(j)).append(' ');
	    }
	    if (lines.size() < countPerClass) {
		lines.add(lineBuilder.toString());
	    } else {
		int index = random.nextInt(lines.size());
		if (index < countPerClass) {
		    lines.set(index, lineBuilder.toString());
		}
	    }
	}
	
	StringBuilder attributeString = new StringBuilder();
	for (Integer attribute : attributes) {
	    attributeString.append(attribute);
	}
	String outputFileFullPrefix = outputDir + File.separatorChar + outputFilePrefix;
	File file = new File(outputFileFullPrefix + "." + attributeString + "_train");
	BufferedWriter writer = new BufferedWriter(new FileWriter(file));
	for (Integer key : classGroup.keySet()) {
	    LinkedList<String> lines = classGroup.get(key);
	    for (String bline : lines) {
		writer.write(bline);
		writer.newLine();
	    }
	}
	writer.close();
	return file;
    }
    
    public static void main1(String[] args) throws Exception {
	args[0] = "/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/expt/txt/shuttle.csv";
	args[1] = "CSV";
	args[2] = "0:[]";
	args[3] = "9";
	args[4] = "/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/expt/txt";
	args[5] = "shuttle_group";
	args[6] = args[4];
	args[7] = args[5];
	
	// Read arguments
	String inputFile = args[0];
	String type = args[1];
	String attributeList = args[2];
	String classIndex = args[3];
	String groupDir = args[4];
	String groupFilePrefix = args[5];
	String trainingDir = args[6];
	String trainingFilePrefix = args[7];
	
	// Read the file
	Instances instances = getInstances(inputFile, type);
	instances.setClassIndex(Integer.parseInt(classIndex));
	Map<Integer, double[]> valueMap = getValueMap(attributeList);
	
	// Pass the data to group by for creating groups
	// This call will change as the group by function evolves
	List<File> groupFiles = GroupBy.doGroup(instances, groupDir, groupFilePrefix, valueMap);
	
	// Create training data
	File trainingFile = createTraining(instances, trainingDir, trainingFilePrefix, new TreeSet<Integer>(valueMap.keySet()), 100);
	
	// Pass the training data to cross validation for parameter selection
	// doCrossValidation
	
	// Pass the training and groups for estimation
	
    }

}
