package revive;

import gnu.trove.list.array.TIntArrayList;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;

public class FormatKDDCUP {
    
    public static void main(String[] args) throws Exception {
	String dataFile = "/mnt/a99/d0/aruniyer/Workspace/kddcup.data_10_percent";
	String outFile = "/mnt/a99/d0/aruniyer/Workspace/kddcup99.dat";
	String namesFile = "/mnt/a99/d0/aruniyer/Workspace/kddcup.names";
	
	TIntArrayList symbolicAttrIndex = new TIntArrayList();
	HashMap<String, Integer> classMap = new HashMap<String, Integer>();
	BufferedReader namesFileReader = new BufferedReader(new FileReader(namesFile));
	String namesFileLine = namesFileReader.readLine(); 
	namesFileLine = namesFileLine.trim();
	namesFileLine = namesFileLine.substring(0, namesFileLine.length() - 1);
	String[] classes = namesFileLine.split(",");
	for (int i = 0; i < classes.length; i++) {
	    classMap.put(classes[i], i);
	}
	for (int i = 0; (namesFileLine = namesFileReader.readLine()) != null; i++) {
	    namesFileLine = namesFileLine.trim();
	    namesFileLine = namesFileLine.substring(0, namesFileLine.length() - 1);
	    String[] cols = namesFileLine.split(":");
	    cols[1] = cols[1].trim();
	    if (cols[1].equalsIgnoreCase("symbolic")) {
		symbolicAttrIndex.add(i);
	    }
	}
	namesFileReader.close();
	
	
	BufferedReader dataFileReader = new BufferedReader(new FileReader(dataFile));
	BufferedWriter dataFileWriter = new BufferedWriter(new FileWriter(outFile));
	String dataLine = null;
	while ((dataLine = dataFileReader.readLine()) != null) {
	    dataLine = dataLine.trim();
	    dataLine = dataLine.substring(0, dataLine.length() - 1);
	    String[] cols = dataLine.split(",");
	    StringBuilder lineBuilder = new StringBuilder();
	    for (int i = 0; i < cols.length - 1; i++) {
		if (!symbolicAttrIndex.contains(i)) {
		    lineBuilder.append(cols[i] + " ");
		}
	    }
	    Integer classVal = classMap.get(cols[cols.length - 1]);
	    if (classVal == null) {
		System.out.println(classMap);
		System.out.println(cols[cols.length - 1]);
		System.exit(0);
	    }
	    lineBuilder.append(classVal);
	    dataFileWriter.write(lineBuilder.toString());
	    dataFileWriter.newLine();
	}
	dataFileReader.close();
	dataFileWriter.close();
    }

}
