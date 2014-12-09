package revive;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;

public class CheckInconsistency {
    
    public static void main(String[] args) throws Exception {
	String inFile = "consistency_output.mmdcr";
	String outFile = "inconsistency_output";
	BufferedReader bufferedReader = new BufferedReader(new FileReader(inFile));
	HashMap<Integer, double[]> predMap = new HashMap<Integer, double[]>();
	HashMap<Integer, Integer> sizeMap = new HashMap<Integer, Integer>();
	HashMap<Integer, String> lineMap = new HashMap<Integer, String>();
	String line = bufferedReader.readLine();
	while ((line = bufferedReader.readLine()) != null) {
	    String[] cols = line.split("\t");
	    String groupID = cols[1];
	    groupID = groupID.substring("SEQ_".length());
	    Integer group = Integer.parseInt(groupID, 2);
	    
	    double[] prop = new double[7];
	    for (int i = 11; i <= 17; i++) {
		prop[i - 11] = Double.parseDouble(cols[i]);
	    }
	    
	    predMap.put(group, prop);
	    sizeMap.put(group, (int) Double.parseDouble(cols[2]));
	    lineMap.put(group, line);
	}
	bufferedReader.close();
	
	BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outFile));
	for (Integer group1 : predMap.keySet()) {
	    double[] pred1 = predMap.get(group1);
	    double size1 = sizeMap.get(group1);
	    for (Integer group2 : predMap.keySet()) {
		double[] pred2 = predMap.get(group2);
		double size2 = sizeMap.get(group2);
		if ((group1 & group2) == 0) {
		    Integer union = group1 | group2;
		    double[] predUnion = predMap.get(union);
		    if (predUnion != null) {
			double sizeUnion = sizeMap.get(union);
			if (!(sizeUnion == size1 + size2))
			    throw new IllegalStateException();
			double max = 0;
			for (int i = 0; i < pred1.length; i++) {
			    double val = (size1 / sizeUnion) * pred1[i] + (size2 / sizeUnion) * pred2[i];
			    max = Math.max(max, Math.abs(val - predUnion[i]));
			}

			if (max >= 0.1) {
			    bufferedWriter.write(lineMap.get(group1));
			    bufferedWriter.newLine();
			    bufferedWriter.write(lineMap.get(group2));
			    bufferedWriter.newLine();
			    bufferedWriter.write(lineMap.get(union));
			    bufferedWriter.newLine();
			}
		    }
		}
	    }
	}
	bufferedWriter.close();
    }

}
