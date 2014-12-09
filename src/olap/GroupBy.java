package olap;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import weka.core.Instance;
import weka.core.Instances;

public class GroupBy {

    public static List<File> doGroup(Instances instances, String outputDir, String outputFilePrefix, Map<Integer, double[]> valueMap) throws Exception {
	HashMap<String, LinkedList<String>> groups = new HashMap<String, LinkedList<String>>();
	TreeSet<Integer> attributes = new TreeSet<Integer>(valueMap.keySet());
	for (int i = 0; i < instances.numInstances(); i++) {
	    Instance instance = instances.instance(i);
	    StringBuilder keyBuilder = new StringBuilder();
	    for (Integer attribute : attributes) {
		double[] values = valueMap.get(attribute);
		if (values == null)
		    keyBuilder.append(instance.value(attribute)).append(',');
		else {
		    double val = instance.value(attribute);
		    int id = -1;
		    for (id = 0; id < values.length; id++)
			if (val < values[id])
			    break;
		    keyBuilder.append(id).append(',');
		}
	    }
	    LinkedList<String> lines = groups.get(keyBuilder.toString());
	    if (lines == null) {
		lines = new LinkedList<String>();
		groups.put(keyBuilder.toString(), lines);
	    }
	    StringBuilder lineBuilder = new StringBuilder();
	    for (int j = 0; j < instances.numAttributes(); j++) {
		if (!attributes.contains(j))
		    lineBuilder.append(instance.value(j)).append(' ');
	    }
	    lines.add(lineBuilder.toString());
	}

	StringBuilder attributeString = new StringBuilder();
	for (Integer attribute : attributes) {
	    attributeString.append(attribute);
	}
	
	LinkedList<File> outFileList = new LinkedList<File>();
	String outputFileFullPrefix = outputDir + File.separatorChar + outputFilePrefix;
	for (String key : groups.keySet()) {
	    File file = new File(outputFileFullPrefix + "." + attributeString + "_" + key);
	    outFileList.add(file);
	    LinkedList<String> lines = groups.get(key);
	    BufferedWriter writer = new BufferedWriter(new FileWriter(file));
	    for (String bline : lines) {
		writer.write(bline);
		writer.newLine();
	    }
	    writer.close();
	}
	return outFileList;
    }

}
