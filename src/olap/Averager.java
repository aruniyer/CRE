package olap;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;

public class Averager {
    
    // Dataset, seed1, seed2, model, scaletype, method, tolerence, trueTheta1, trueTheta2, theta1, theta2, inInterval, pr, interval
    // unique upto dataset,model,scaletype,method,tolerance, trueTheta1, trueTheta2
    // average everything else
    public static void main(String[] args) throws Exception {
	String filename = "/mnt/a99/d0/aruniyer/Workspace/Java/InfoLab/TDA/expt/olap_diff.tsv"; // args[0];
	
	HashMap<String, Double> totalInIntervalMap = new HashMap<String, Double>();
	HashMap<String, Double> totalPrMap = new HashMap<String, Double>();
	HashMap<String, Double> totalTheta1Map = new HashMap<String, Double>();
	HashMap<String, Double> totalTheta2Map = new HashMap<String, Double>();
	HashMap<String, Integer> countMap = new HashMap<String, Integer>();
	int DATASET = 0;
	int MODEL = 3;
	int SCALETYPE = 4;
	int METHOD = 5;
	int TOLERANCE = 6;
	int TRUETHETA1 = 7;
	int TRUETHETA2 = 8;
	int THETA1 = 9;
	int THETA2 = 10;
	int ININTERVAL = 11;
	int PR = 12;
	
	int[] keyIndex = {DATASET,MODEL,SCALETYPE,METHOD,TOLERANCE,TRUETHETA1, TRUETHETA2};
	BufferedReader reader = new BufferedReader(new FileReader(filename));
	String line = reader.readLine(); // remove header
	int counter = 1;
	while ((line = reader.readLine()) != null) {
	    if (counter % 10000 == 0)
		System.out.println(counter);
	    counter++;
	    String[] fields = line.split("\t");
	    String key = "";
	    for (int index : keyIndex) {
		key = key + "\t" + fields[index];
	    }
	    key = key.trim();
	    
	    doIncrement(totalInIntervalMap, key, Double.valueOf(fields[ININTERVAL]));
	    doIncrement(totalPrMap, key, Double.valueOf(fields[PR]));
	    doIncrement(totalTheta1Map, key, Double.valueOf(fields[THETA1]));
	    doIncrement(totalTheta2Map, key, Double.valueOf(fields[THETA2]));
	    Integer count = countMap.get(key);
	    if (count == null)
		count = 0;
	    countMap.put(key, count + 1);
	}
	reader.close();
	
	BufferedWriter averagedVersion = new BufferedWriter(new FileWriter(filename + ".avg"));
	averagedVersion.write("dataset\t model\t scaletype\t method\t tolerance\t trueTheta1\t trueTheta2\t avgTheta1\t avgTheta2\t avgInInterval\t avgPr");
	averagedVersion.newLine();
	for (String key : countMap.keySet()) {
	    Double inInterval = totalInIntervalMap.get(key);
	    Double pr = totalPrMap.get(key);
	    Double theta1 = totalTheta1Map.get(key);
	    Double theta2 = totalTheta2Map.get(key);
	    Integer count = countMap.get(key);
	    double avgInInterval = inInterval / (double) count;
	    double avgPr = pr / (double) count;
	    double avgTheta1 = theta1 / (double) count;
	    double avgTheta2 = theta2 / (double) count;
	    averagedVersion.write(key + "\t" + avgTheta1 + "\t" + avgTheta2 + "\t" + avgInInterval + "\t" + avgPr);
	    averagedVersion.newLine();
	}
	averagedVersion.close();
    }
    
    public static void doIncrement(HashMap<String, Double> map, String key, Double inc) {
	Double val = map.get(key);
	if (val == null)
	    val = 0.0;
	map.put(key, val + inc);
    }

}
