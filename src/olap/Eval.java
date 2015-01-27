package olap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Constructor;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.SimpsonIntegrator;
import org.apache.commons.math3.analysis.integration.UnivariateIntegrator;
import org.apache.commons.math3.distribution.AbstractRealDistribution;
import org.apache.commons.math3.distribution.BetaDistribution;

public class Eval {

	public static class Interval {
		double lb;
		double ub;
		double pr;

		boolean contains(double val) {
			return (lb <= val && val <= ub);
		}

		public double width() {
			return (ub - lb);
		}
	}

	public Interval outputIntervals(double cdfs[], double tolerence,
			double aroundPt, double startPt, double endPt) {
		Interval interval = new Interval();

		double inc = (endPt - startPt) / cdfs.length;
		for (double x0 = startPt; x0 < endPt - tolerence; x0 += inc) {
			if (aroundPt > startPt
					&& (x0 > aroundPt || aroundPt > x0 + tolerence))
				continue;
			double pr = getCDF(cdfs, inc, x0 + tolerence, startPt)
					- getCDF(cdfs, inc, x0, startPt);
			if (interval.pr < pr) {
				interval.pr = pr;
				interval.lb = x0;
				interval.ub = x0 + tolerence;
			}
		}
		return interval;
	}

	private double getCDF(double[] cdfs, double inc, double d, double startPt) {
		return cdfs[(int) ((d - startPt) / inc)];
	}

	private Interval outputIntervalsFixedPr(double cdfs[], double prThresholds,
			double aroundPt, double startPt, double endPt) {
		int i = 0;
		Interval interval = new Interval();
		interval.lb = startPt;
		interval.ub = endPt;
		interval.pr = 1;
		double inc = (endPt - startPt) / (cdfs.length - 1);
		for (double x0 = startPt; x0 < endPt; x0 += inc, i++) {
			int startJ = -1;
			for (int j = 0; j < i; j++) {
				if (cdfs[i] - cdfs[j] < prThresholds)
					break;
				startJ = j;
			}
			if (startJ < 0)
				continue;
			if (aroundPt > startPt
					&& (aroundPt < startJ * inc || aroundPt > i * inc))
				continue;
			if ((i - startJ) * inc < interval.width()) {
				interval.lb = startPt + startJ * inc;
				interval.ub = x0;
				interval.pr = prThresholds;
			}
		}
		return interval;
	}

	private double[] cacheCDFs(AbstractRealDistribution dist, double startPt,
			double endPt) {
		double cdfs[] = new double[20001];
		double inc = (endPt - startPt) / (cdfs.length - 1);
		int i = 0;
		for (double x0 = startPt; x0 < endPt; x0 += inc, i++) {
			cdfs[i] = dist.cumulativeProbability(x0 + inc);
		}
		return cdfs;
	}

	public void processRecord(String label, double trueTheta, double theta,
			double scale) {
		String methods[] = { "mmdIsMean", "mmdIsMeanInt", "mmdIsMod" };
		double tolerence[] = new double[] { 0.01, 0.05, 0.1, 0.15, 0.2 };
		double conf[] = new double[] { 0.99, 0.95, 0.9, 0.8, 0.7 };
		for (int m = 0; m < methods.length; m++) {
			double[] mn = getmn(theta, scale, m);
			BetaDistribution betaDist = new BetaDistribution(mn[0], mn[1]);
			double cdfs[] = cacheCDFs(betaDist, 0, 1);
			for (double tol : tolerence) {
				Interval interval = outputIntervals(cdfs, tol, m == 1 ? theta
						: -1, 0, 1);
				System.out.println(label + "\t" + methods[m] + "\t" + tol
						+ "\t" + trueTheta + "\t" + theta + "\t"
						+ (interval.contains(trueTheta) ? 1 : 0) + "\t"
						+ interval.pr + "\t" + interval.lb);
			}
			for (double pr : conf) {
				Interval interval = outputIntervalsFixedPr(cdfs, pr,
						m == 1 ? theta : -1, 0, 1);
				System.out.println(label + "\t" + methods[m] + "Pr\t"
						+ interval.width() + "\t" + trueTheta + "\t" + theta
						+ "\t" + (interval.contains(trueTheta) ? 1 : 0) + "\t"
						+ interval.pr + "\t" + interval.lb);
			}
		}
	}

	public enum SCALETYPES {
		ThetaMMDAsMean {
			public String getFileNick() {
				return "mean";
			}
		},
		ThetaMeanAsMean {
			public String getFileNick() {
				return "mode1";
			}
		},
		ThetaMMDAsMode {
			public String getFileNick() {
				return "mode2";
			}
		};

		public abstract String getFileNick();
	}

	private static int TRUE_THETA = 0;
	private static int THETA_MMD = 2;
	private static int SCALE = 5;

	public void processFileModel1(File fileName, SCALETYPES pf)
			throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String labToks[] = fileName.getName().split("_");
		String line = br.readLine();
		while ((line = br.readLine()) != null) {
			String flds[] = line.split(",");
			processRecord(labToks[2] + "\t" + labToks[4] + "\tModel1\t" + pf,
					Double.parseDouble(flds[TRUE_THETA]),
					Double.parseDouble(flds[THETA_MMD]),
					Double.parseDouble(flds[SCALE]));
		}
	}

	public static void mainModel1(SCALETYPES pf) throws IOException {
		if (pf == SCALETYPES.ThetaMMDAsMean)
			SCALE = 5;
		else if (pf == SCALETYPES.ThetaMeanAsMean)
			SCALE = 6;
		else if (pf == SCALETYPES.ThetaMMDAsMode)
			SCALE = 7;
		Eval eval = new Eval();
		File file = new File(
				"/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/betaregfeatures/");
		String pattern = "20140510_feature_";
		for (File fname : file.listFiles()) {
			if (!fname.getName().startsWith(pattern))
				continue;
			if (!fname.getName().contains("test"))
				continue;
			eval.processFileModel1(fname, pf);
		}
	}

	public void processFileModel2(File fileName, SCALETYPES pf)
			throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String line = null;
		while ((line = br.readLine()) != null) {
			if (line.contains("BetaregRMode"))
				continue;
			String flds[] = line.split(",");
			String dataset = flds[0].toLowerCase();
			String seed = flds[16];
			processRecord(dataset + "\t" + seed + "\tModel2\t" + pf,
					Double.parseDouble(flds[9]), Double.parseDouble(flds[11]),
					Double.parseDouble(flds[12]));
		}
	}

	public static void mainModel2(SCALETYPES pf) throws IOException {
		Eval eval = new Eval();
		File file = new File(
				"/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/betaregfeatures/20140510_"
						+ pf.getFileNick() + "_results");
		eval.processFileModel2(file, pf);
	}

	public static void main(String[] args) throws Exception {
		// System.setOut(new PrintStream("expt/olap_nondiff.txt"));
		// System.out.println("Dataset\t seed\t model\t scaletype\t method\t tolerence \t trueTheta \t theta \t inInterval \t pr \t interval");
		// SCALETYPES pf = SCALETYPES.ThetaMMDAsMean;
		// for (SCALETYPES pf : SCALETYPES.values()) {
		// mainModel1(pf);
		// mainModel2(pf);
		// }

		String model = args[0];
		SCALETYPES pf = SCALETYPES.valueOf(args[1]);
		String dataset = args[2];
		String function = args[3];
		System.setOut(new PrintStream("expt/olap_" + function + "_" + model
				+ "_" + pf.getFileNick() + "_" + dataset + ".txt"));
		System.out
				.println("Dataset\t seed1\t seed2\t model\t scaletype\t method\t tolerence\t trueTheta1\t trueTheta2\t theta1\t theta2\t inInterval\t pr\t interval");
		// for (SCALETYPES pf : SCALETYPES.values()) {
		// for (String model : new String[]{"Model1", "Model2"}) {
		// System.setOut(new PrintStream("expt/olap_diff_" + model + "_" +
		// pf.getFileNick() + ".txt"));
		// System.out.println("Dataset\t seed\t model\t scaletype\t method\t tolerence\t trueTheta1\t trueTheta2\t theta1\t theta2\t inInterval\t pr\t interval");
		// mainDiff(args, model, pf);
		// }
		// }
		if (function.equalsIgnoreCase("diff"))
			mainFn(dataset, model, pf,
					DiffBetaDistribution.class.getCanonicalName());
		else if (function.equalsIgnoreCase("ratio"))
			mainFn(dataset, model, pf,
					RatioBetaDistribution.class.getCanonicalName());
	}

	public static void mainFn(String dataset, String model, SCALETYPES pf,
			String function) throws Exception {
		Eval eval = new Eval();
		File file = null;
		if (model.equalsIgnoreCase("model2"))
			file = new File(
					"/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/betaregfeatures/20140510_diff_"
							+ pf.getFileNick() + "_model2_results");
		else
			file = new File(
					"/mnt/bag/wwt/YoutubeCommentsDataset/SAUCe/betaregfeatures/20140510_diff_"
							+ pf.getFileNick() + "_model1_results");
		eval.processFileFn(dataset, file, model, pf, function);
	}

	public static boolean debug = false;

	public void processFileFn(String dtset, File fileName, String model,
			SCALETYPES pf, String function) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String line = null;
		while ((line = br.readLine()) != null) {
			if (line.contains("BetaMeanDiff")) {
				String flds[] = line.split(",");
				String dataset = flds[0].toLowerCase();
				if (!dataset.equalsIgnoreCase(dtset)) {
					continue;
				}
				String seed1 = flds[1];
				String seed2 = flds[2];
				// if (flds[3].equalsIgnoreCase("0.1") &&
				// flds[4].equalsIgnoreCase("0.9")) {
				double trTheta1 = Double.parseDouble(flds[3]);
				double trTheta2 = Double.parseDouble(flds[4]);
				processRecord(dataset + "\t" + seed1 + "\t" + seed2 + "\t"
						+ model + "\t" + pf, trTheta1, trTheta2,
						Double.parseDouble(flds[5]),
						Double.parseDouble(flds[6]),
						Double.parseDouble(flds[7]),
						Double.parseDouble(flds[8]), function);
				// }
			}
		}
	}

	public double[] getmn(double theta, double scale, int m) {
		double[] mn = new double[2];
		scale = Math.max(scale, 3);
		double mean = (m <= 1) ? theta : (theta * (scale - 2) + 1) / scale;
		assert (mean <= 1 && mean >= 0);
		if (mean < 1 && mean > 0)
			scale = Math.max(Math.max(scale, 1 / mean), 1 / (1 - mean));
		else if (mean == 0)
			mean = 1 / scale;
		else if (mean == 1)
			mean = 1 - (1 / scale);
		mn[0] = mean * scale;
		mn[1] = (1 - mean) * scale;
		return mn;
	}

	public void processRecord(String label, double trueTheta1,
			double trueTheta2, double theta1, double scale1, double theta2,
			double scale2, String function) throws Exception {
		@SuppressWarnings("rawtypes")
		Constructor constructor = Class.forName(function).getConstructor(
				Double.TYPE, Double.TYPE, Double.TYPE, Double.TYPE);
		String methods[] = { "mmdIsMean", "mmdIsMeanInt", "mmdIsMod" };
		double tolerenceDiff[] = new double[] { 0.01, 0.05, 0.1, 0.15, 0.2 };
		double tolerenceRatio[] = new double[] { 0.5, 0.1, 1.5, 2, 2.5 };
		double conf[] = new double[] { 0.99, 0.95, 0.9, 0.8, 0.7 };
		for (int m = 0; m < methods.length; m++) {
			double[] mn1 = getmn(theta1, scale1, m);
			double[] mn2 = getmn(theta2, scale2, m);
			AbstractRealDistribution fnBetaDist = (AbstractRealDistribution) constructor
					.newInstance(mn1[0], mn1[1], mn2[0], mn2[1]);
			double cdfs[] = null;
			if (function.equalsIgnoreCase(DiffBetaDistribution.class
					.getCanonicalName())) {
				cdfs = cacheCDFs(fnBetaDist, -1, 1);
				for (double tol : tolerenceDiff) {
					Interval interval = outputIntervals(cdfs, tol,
							m == 1 ? (theta1 - theta2) : -1, -1, 1);
					System.out.println(label
							+ "\t"
							+ methods[m]
							+ "\t"
							+ tol
							+ "\t"
							+ trueTheta1
							+ "\t"
							+ trueTheta2
							+ "\t"
							+ theta1
							+ "\t"
							+ theta2
							+ "\t"
							+ (interval.contains(trueTheta1 - trueTheta2) ? 1
									: 0) + "\t" + interval.pr + "\t"
							+ interval.lb);
				}
				for (double pr : conf) {
					Interval interval = outputIntervalsFixedPr(cdfs, pr,
							m == 1 ? (theta1 - theta2) : -1, -1, 1);
					System.out.println(label
							+ "\t"
							+ methods[m]
							+ "Pr\t"
							+ interval.width()
							+ "\t"
							+ trueTheta1
							+ "\t"
							+ trueTheta2
							+ "\t"
							+ theta1
							+ "\t"
							+ theta2
							+ "\t"
							+ (interval.contains(trueTheta1 - trueTheta2) ? 1
									: 0) + "\t" + interval.pr + "\t"
							+ interval.lb);
				}
			} else {
				cdfs = cacheCDFs(fnBetaDist, 0, 1000);
				if (theta2 <= 1e-10)
					theta2 = 1e-10;
				for (double tol : tolerenceRatio) {
					Interval interval = outputIntervals(cdfs, tol,
							m == 1 ? (theta1 / theta2) : -1, 0, 1000);
					System.out.println(label
							+ "\t"
							+ methods[m]
							+ "\t"
							+ tol
							+ "\t"
							+ trueTheta1
							+ "\t"
							+ trueTheta2
							+ "\t"
							+ theta1
							+ "\t"
							+ theta2
							+ "\t"
							+ (interval.contains(trueTheta1 / trueTheta2) ? 1
									: 0) + "\t" + interval.pr + "\t"
							+ interval.lb);
				}
				for (double pr : conf) {
					Interval interval = outputIntervalsFixedPr(cdfs, pr,
							m == 1 ? (theta1 / theta2) : -1, 0, 1000);
					System.out.println(label
							+ "\t"
							+ methods[m]
							+ "Pr\t"
							+ interval.width()
							+ "\t"
							+ trueTheta1
							+ "\t"
							+ trueTheta2
							+ "\t"
							+ theta1
							+ "\t"
							+ theta2
							+ "\t"
							+ (interval.contains(trueTheta1 / trueTheta2) ? 1
									: 0) + "\t" + interval.pr + "\t"
							+ interval.lb);
				}
			}

		}
	}

	public static void getch() {
		try {
			System.in.read();
		} catch (Exception e) {

		}
	}

}

@SuppressWarnings("deprecation")
abstract class TwoBetaFunctionDistribution extends AbstractRealDistribution {

	private static final long serialVersionUID = 1119387002120391294L;

	protected static int MAX_EVAL = Integer.MAX_VALUE;
	protected UnivariateIntegrator integrator = new SimpsonIntegrator(1e-4,
			1e-4, SimpsonIntegrator.DEFAULT_MIN_ITERATIONS_COUNT,
			SimpsonIntegrator.SIMPSON_MAX_ITERATIONS_COUNT);
	protected BetaDistribution beta1, beta2;
	protected double totalArea;
	protected int cachepdflen = 20000;
	protected double[] cachepdf = new double[cachepdflen + 1];

	protected double[] rescale(double m, double n) {
		double[] mn = new double[2];
		if (m > 1000 || n > 1000) {
			double min = Math.min(m, n);
			m = m / min;
			n = n / min;
		}
		if (m > 1000) {
			m = 1000;
		} else if (n > 1000) {
			n = 1000;
		}
		mn[0] = m;
		mn[1] = n;
		return mn;
	}

}

class DiffBetaDistribution extends TwoBetaFunctionDistribution {

	private static final long serialVersionUID = 70951874571610100L;

	private DiffBetaCore dbc = new DiffBetaCore();
	private double inc = 2.0 / (double) cachepdflen;

	private class DiffBetaCore implements UnivariateFunction {

		double temp_d;

		@Override
		public double value(double t) {
			double lhs = getBetaDensity(t + temp_d, beta1);
			double rhs = getBetaDensity(t, beta2);
			return lhs * rhs;
		}

		private double getBetaDensity(double x, BetaDistribution beta) {
			if (isCloseTo(x, 0) && isCloseTo(beta.getAlpha(), 1))
				return beta.getBeta();
			else if (isCloseTo(x, 1) && isCloseTo(beta.getBeta(), 1))
				return beta.getAlpha();
			else {
				return beta.density(x);
			}
		}

		private boolean isCloseTo(double x1, double target) {
			double eps = 1e-6;
			if (target - eps <= x1 && x1 <= target + eps) {
				return true;
			}
			return false;
		}

	}

	public DiffBetaDistribution(final double m1, final double n1,
			final double m2, final double n2) {
		double[] mn = rescale(m1, n1);
		beta1 = new BetaDistribution(mn[0], mn[1]);
		mn = rescale(m2, n2);
		beta2 = new BetaDistribution(mn[0], mn[1]);

		totalArea = 0;
		int i = 0;
		for (double x = -1; x <= 1; x += inc) {
			cachepdf[i] = density(x);
			totalArea += cachepdf[i];
			i++;
		}
	}

	@Override
	public double cumulativeProbability(double x) {
		double area = 0;
		int i = 0;
		for (double x_0 = -1; x_0 < x; x_0 += inc, i++)
			area += cachepdf[i];
		return area / totalArea;
	}

	@Override
	public double density(double d) {
		dbc.temp_d = d;
		if (d < 0)
			if (d > -1)
				return integrator.integrate(MAX_EVAL, dbc, -d, 1);
			else
				return 0;
		else if (d < 1)
			return integrator.integrate(MAX_EVAL, dbc, 0, 1 - d);
		else
			return 0;
	}

	@Override
	public double getNumericalMean() {
		return beta1.getNumericalMean() - beta2.getNumericalMean();
	}

	@Override
	public double getNumericalVariance() {
		return beta1.getNumericalVariance() + beta2.getNumericalVariance();
	}

	@Override
	public double getSupportLowerBound() {
		return -1;
	}

	@Override
	public double getSupportUpperBound() {
		return 1;
	}

	@Override
	public boolean isSupportConnected() {
		return true;
	}

	@Override
	public boolean isSupportLowerBoundInclusive() {
		return true;
	}

	@Override
	public boolean isSupportUpperBoundInclusive() {
		return true;
	}

}

class RatioBetaDistribution extends TwoBetaFunctionDistribution {

	private static final long serialVersionUID = -1561359702582082045L;
	private RatioBetaCore rbc = new RatioBetaCore();
	private double inc = 1000.0 / (double) cachepdflen;

	private class RatioBetaCore implements UnivariateFunction {

		double temp_r;

		@Override
		public double value(double t) {
			double lhs = getBetaDensity(t * temp_r, beta1);
			double rhs = getBetaDensity(t, beta2);
			return lhs * rhs;
		}

		private double getBetaDensity(double x, BetaDistribution beta) {
			if (isCloseTo(x, 0) && isCloseTo(beta.getAlpha(), 1))
				return beta.getBeta();
			else if (isCloseTo(x, 1) && isCloseTo(beta.getBeta(), 1))
				return beta.getAlpha();
			else {
				return beta.density(x);
			}
		}

		private boolean isCloseTo(double x1, double target) {
			double eps = 1e-6;
			if (target - eps <= x1 && x1 <= target + eps) {
				return true;
			}
			return false;
		}

	}

	public RatioBetaDistribution(final double m1, final double n1,
			final double m2, final double n2) {
		double[] mn = rescale(m1, n1);
		beta1 = new BetaDistribution(mn[0], mn[1]);
		mn = rescale(m2, n2);
		beta2 = new BetaDistribution(mn[0], mn[1]);

		totalArea = 0;
		int i = 0;
		for (double x = 0; x <= 1000; x += inc) {
			cachepdf[i] = density(x);
			totalArea += cachepdf[i];
			i++;
		}
	}

	@Override
	public double cumulativeProbability(double x) {
		double area = 0;
		int i = 0;
		for (double x_0 = 0; x_0 < x; x_0 += inc, i++)
			area += cachepdf[i];
		return area / totalArea;
	}

	@Override
	public double density(double r) {
		rbc.temp_r = r;
		if (r < 0)
			throw new IllegalStateException();
		return integrator.integrate(MAX_EVAL, rbc, 0, Math.min(1, 1 / r));
	}

	@Override
	public double getNumericalMean() {
		double ex = beta1.getNumericalMean();
		double ey = beta2.getNumericalMean();
		double vary = beta2.getNumericalVariance();
		return (ex / ey) + (vary * ex / (ey * ey * ey));
	}

	@Override
	public double getNumericalVariance() {
		double ex = beta1.getNumericalMean();
		double varx = beta1.getNumericalVariance();
		double ey = beta2.getNumericalMean();
		double vary = beta2.getNumericalVariance();
		return (varx / (ey * ey)) + ((ex * ex * vary) / (ey * ey * ey * ey));
	}

	@Override
	public double getSupportLowerBound() {
		return 0;
	}

	@Override
	public double getSupportUpperBound() {
		return Double.POSITIVE_INFINITY;
	}

	@Override
	public boolean isSupportLowerBoundInclusive() {
		return true;
	}

	@Override
	public boolean isSupportUpperBoundInclusive() {
		return false;
	}

	@Override
	public boolean isSupportConnected() {
		return true;
	}

}