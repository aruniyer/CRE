package revive;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

public class GenerateDataHandlerTest {

    public static void main(String[] args) throws Exception {
	BufferedWriter writer = new BufferedWriter(new FileWriter("groupbytest_test.dat"));
	double yprob = 0.2;
	
	int[] limits = { 3, 3, 3, 3 };
	double[] means = {-0.5, 0, 0.5};
	Random random = new Random(1);
	int[] counter = new int[limits.length];
	double[] val = new double[limits.length];
	
	for (counter[0] = 0; counter[0] < limits[0]; counter[0]++) {
	    for (counter[1] = 0; counter[1] < limits[1]; counter[1]++) {
		for (counter[2] = 0; counter[2] < limits[2]; counter[2]++) {
		    for (counter[3] = 0; counter[3] < limits[3]; counter[3]++) {
			for (int i = 0; i < val.length; i++) {
//			    double mean = means[counter[i]];
//			    val[i] = random.nextGaussian()*0.5 + mean;
			    val[i] = means[counter[i]];
			    writer.write(val[i] + " ");
			}
			if (random.nextDouble() <= yprob)
			    writer.write("0");
			else
			    writer.write("1");
			writer.newLine();
		    }
		}
	    }
	}
	writer.close();
    }
    
}
