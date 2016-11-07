/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tsc_algorithms.boss;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import utilities.Timer;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;

/**
 * Bag of Temporal SIFT Words classifier to be used with known parameters, for botsw with parameter search, use BoTSWEnsemble.
 * 
 * Will use euclidean distance by default. If svm wanted, call setUseSVM(true). Precise SVM implementation/accuracy could not be recreated, likewise 
 * for kmeans, epsilon value ignored
 * 
 * Params: n_b, a, k, c_svm(if using svm)
 * 
 * @author James Large
 * 
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BoTSW implements Classifier, Serializable, SaveCVAccuracy{

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Bailly, Adeline and Malinowski, Simon and Tavenard, Romain and Guyet, Thomas and Chapel, Laetitia");
        result.setValue(TechnicalInformation.Field.TITLE, "Bag-of-Temporal-SIFT-Words for Time Series Classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");

        return result;
    }
    
    
    public SimpleKMeans kmeans = null;
    public LibSVM svm = null;
    public Params params;
    public BoTSW_Bag[] bags;
    public FeatureDiscoveryData[] fdData;
    private boolean useSVM = false;
    private boolean clusteringDataPreBuilt = false;
    private boolean preprocessDataPreBuilt = false;
    private boolean searchingForK = false;
    
    //keeping these as members so that when header info is found in training, can be reused for testing
    public Instances clusterData;
    public Instances bagData; 
    
    private String trainCVPath; //SaveCVAccuracy
    private boolean trainCV=false;
    
    public BoTSW() { 
        params = new Params(4, 4, 32, 1); //default smallest of each 
    }
    
    public BoTSW(int n_b, int a, int k) {
        params = new Params(n_b, a, k);
    }
    
    public BoTSW(int n_b, int a, int k, int c_svm) {
        params = new Params(n_b, a, k, c_svm);
    }

    @Override
    public void setCVPath(String train) {
        trainCVPath=train;
        trainCV=true;
    }

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("n_b=").append(params.n_b).append("/a=").append(params.a);
        sb.append("/k=").append(params.k);

        return sb.toString();
    }
    
    public static class Params {
        //set (or searched for) by user)
        public int n_b;
        public int a; 
        public int k;
        public double c_svm;
        
        public int denseSampleRate = 0;
        public int n_sc; //calculated via series length
        
        //fixed
        public final double k_sc = 1.257013374521;
        public final double sigma = 1.6;
        //normalise booleans ommitted, assuming SSR/l2 norm
        
        public Params(int n_b, int a, int k) {
            this.n_b = n_b;
            this.a = a;
            this.k = k;
            this.c_svm = 1;
            this.n_sc = 0;
        }
        
        public Params(int n_b, int a, int k, double c_svm) {
            this.n_b = n_b;
            this.a = a;
            this.k = k;
            this.c_svm = c_svm;
            this.n_sc = 0;
        }
        
        public Params(int n_b, int a, int k, double c_svm, int n_sc) {
            this.n_b = n_b;
            this.a = a;
            this.k = k;
            this.c_svm = c_svm;
            this.n_sc = n_sc;
        }
        
        public int calcNumScales(int seriesLength) {
            int max_sc = (int)(Math.log(0.125 * seriesLength / sigma) / Math.log(k_sc));
            if(n_sc == 0 || n_sc > max_sc)
                n_sc = max_sc;
            return n_sc;
        }
        
        public int calcDenseSampleRate(int seriesLength) {
//            denseSampleRate = (int)((Math.log(seriesLength-200)) / 2);
            denseSampleRate = seriesLength / 100;
            if (denseSampleRate < 1) denseSampleRate = 1;
            return denseSampleRate;
        }
        
        public void setDenseSampleRate(int rate) {
            denseSampleRate = rate;
        }
        
        @Override
        public String toString() {
            return n_b + "_" + a + "_" + k;
        }
    }
    
    public static class FeatureDiscoveryData {
        public ArrayList<KeyPoint> keypoints;
        public GuassianData gdata;
        
        public FeatureDiscoveryData(ArrayList<KeyPoint> keypoints, GuassianData gdata) {
            this.keypoints = keypoints;
            this.gdata = gdata;
        }
    }
    
    public static class GuassianData {
        public double[][] guassSeries; 
        public double[][] DoGs;
    }
    
    public static class KeyPoint {
        public int time; 
        public int scale; 
        
        public KeyPoint(int time, int scale) {
            this.time = time;
            this.scale = scale;
        }
    }
    
    public static class BoTSW_Bag {
        double[] hist;
        double classValue;
        
        public BoTSW_Bag(double[] hist, double classValue) {
            this.hist = hist;
            this.classValue = classValue;
        }
    }    
    
    public void setSearchingForK(boolean b) {
        searchingForK = b;
    }
    
    public void setUseSVM(boolean use) {
        useSVM = use;
    }
    
    public int[] getParametersValues() {
        return new int[] {params.n_b, params.a, params.k};
    }
    
    public void giveFeatureDiscoveryData(FeatureDiscoveryData[] data) {
        fdData = data;
        preprocessDataPreBuilt = true;
    }
    
    public void giveFeatureData(Instances features) {
        clusterData = features;
        clusteringDataPreBuilt = true;
        preprocessDataPreBuilt = true;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {      
        data = new Instances(data);
        
        if (params.n_sc == 0) //has not already been set
            params.calcNumScales(data.numAttributes()-1);
        
        if (params.denseSampleRate == 0)
            params.calcDenseSampleRate(data.numAttributes()-1);
        
        if (!preprocessDataPreBuilt) { 
            //build the guassian and keypoint location data if not already built
            fdData = new FeatureDiscoveryData[data.numInstances()];
            
            for (int i = 0; i < data.numInstances(); ++i) {
                GuassianData gdata = findDoGs(toArrayNoClass(data.get(i)), params.sigma, params.k_sc, params.n_sc);
                ArrayList<KeyPoint> keypoints = findDenseKeypoints(gdata);
                fdData[i] = new FeatureDiscoveryData(keypoints, gdata);
            }
        }   
        
        if (!clusteringDataPreBuilt) { 
            //describe the keypoints found before using the current parameter settings 
            //n_b and a, may have already been found during parameter search if now just 
            //searching for values of k in the clustering or c_svm (if using svm)
            double[][][] features = new double[data.numInstances()][][];   
            for (int i = 0; i < data.numInstances(); ++i)
                features[i] = describeKeyPoints(fdData[i].gdata.guassSeries, fdData[i].keypoints);
            
            //stuff features into instances format
            FastVector<Attribute> atts = new FastVector<>();
            assert(features[0][0].length == params.n_b*2);
            for (int i = 0; i < features[0][0].length; ++i)
                atts.add(new Attribute(""+i));
            clusterData = new Instances("ClusterInfo", atts, features.length * features[0].length);
            for (int i = 0; i < features.length; ++i)
                for (int j = 0; j < features[i].length; ++j)
                    clusterData.add(new DenseInstance(1, features[i][j]));
        }
        
        //cluster the feature descriptions that have been found/provided
        
        Timer clusteringTimer = new Timer("\t\t\tclustering: n_b="+params.n_b+" a="+params.a+" k="+params.k);
        
        int maxIterations = 10;
        int numAttempts = searchingForK ? 2 : 10;
        double epsilon = 0.001; //not used atm, simplekmeans does not support/non trivial to implement
        double bestCompactness = Double.MAX_VALUE; //custom implemented
        
        for (int i = 0; i < numAttempts; ++i) {
            
            SimpleKMeans t_kmeans = new SimpleKMeans();
            t_kmeans.setMaxIterations(maxIterations);
            t_kmeans.setInitializeUsingKMeansPlusPlusMethod(true);
            t_kmeans.setSeed(i);
            t_kmeans.setNumClusters(params.k);
            t_kmeans.setPreserveInstancesOrder(true); //needed to call .getAssignments()
            t_kmeans.buildClusterer(clusterData);
            
            if (numAttempts > 1) {
                double compactness = compactnessOfClustering(t_kmeans, clusterData);
                if (compactness < bestCompactness)
                    kmeans = t_kmeans;
            } 
            else
                kmeans = t_kmeans;
        }
        
        //code for single attempt of x iterations 
//        int maxIterations = 100;
//        kmeans = new SimpleKMeans();
//        kmeans.setMaxIterations(maxIterations);
//        kmeans.setInitializeUsingKMeansPlusPlusMethod(true);
//        kmeans.setNumClusters(params.k);
//        kmeans.setPreserveInstancesOrder(true); //needed to call .getAssignments()
//        kmeans.buildClusterer(clusterData);
        
        int [] assignments = kmeans.getAssignments(); //final assignments of each FEATURE
        
        //build histograms
        bags = new BoTSW_Bag[data.numInstances()];
        
        int featsPerSeries = clusterData.numInstances() / data.numInstances();
        int feat = 0;
        for (int i = 0; i < data.numInstances(); ++i) {
            double[] hist = new double[params.k];
            for (int j = 0; j < featsPerSeries; ++j)
                ++hist[assignments[feat++]];
            
            hist = normaliseHistogramSSR(hist);
            hist = normaliseHistograml2(hist);
            bags[i] = new BoTSW_Bag(hist, data.get(i).classValue());
        }
        
        //CODE FOR USING SVM FOR CORRECTNESS TESTING PURPOSES, WASNT REALLY WOKRING (DISCREPANCIES IN SVM IMPLEMENTATIONS)
        if (useSVM) {
            Timer svmTimer = new Timer("\t\t\ttrainingsvm");
            
            //stuff back into instances
            FastVector<Attribute> bagatts = new FastVector<>();
            for (int i = 0; i < params.k; ++i)
                bagatts.add(new Attribute(""+i));
            
            List<String> classVals = new ArrayList<>(data.numClasses());
            for (int i = 0; i < data.numClasses(); ++i)
                classVals.add(""+i);
            bagatts.add(new Attribute("classVal", classVals));
            
            bagData = new Instances("Bags", bagatts, data.numInstances());
            bagData.setClassIndex(bagData.numAttributes()-1);
            for (int i = 0; i < bags.length; ++i) {
                double[] inst = new double[params.k+1];
                for (int j = 0; j < params.k; ++j) {
                    inst[j] = bags[i].hist[j];
                }
                inst[inst.length-1] = bags[i].classValue;
                bagData.add(new DenseInstance(1, inst));
            }
            
            //train svm, as close to original in-code params as i can seem to get
            svm = new LibSVM();
            svm.setCost(params.c_svm); 
            svm.setCoef0(0);
            svm.setEps(0.001);
            svm.setGamma(0.5);
            svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
            svm.setDegree(3);
            svm.setNu(0.5);
            svm.setShrinking(true);
            svm.setCacheSize(200);
            svm.setProbabilityEstimates(false);
           
            svm.buildClassifier(bagData);
            
            svmTimer.printlnTimeSoFar();
        }
    }
    
    public static double compactnessOfClustering(SimpleKMeans kmeans, Instances input) throws Exception {
        Instances centroids = kmeans.getClusterCentroids();
        int[] assignments = kmeans.getAssignments();
        
        double totalSqDist = 0.0;
        for (int i = 0; i < assignments.length; ++i) {
            Instance sample = input.get(i);
            Instance centroid = centroids.get(assignments[i]);
            
            for (int j = 0; j < sample.numAttributes(); ++j)
                totalSqDist += (sample.value(j) - centroid.value(j)) * (sample.value(j) - centroid.value(j));
        }
        return totalSqDist;
    }
    
    double[] normaliseHistograml2(double[] hist) {
        double n = 0.;
        for(int x=0; x<hist.length; ++x)
            n += hist[x] * hist[x];
        for(int i = 0; i < hist.length; ++i)
            hist[i] /= n;

        return hist;
    }
    
    double[] normaliseHistogramSSR(double[] hist) {
        for (int j = 0; j < hist.length; ++j)
            hist[j] = Math.sqrt(hist[j]);
        return hist;
    }
    
    double[][] extractFeatures(double[] series) throws Exception {
        GuassianData gdata = findDoGs(series, params.sigma, params.k_sc, params.n_sc);
        ArrayList<KeyPoint> keypoints = findDenseKeypoints(gdata);
        return describeKeyPoints(gdata.guassSeries, keypoints);
    }
    
    double[][] describeKeyPoints(double[][] series, ArrayList<KeyPoint> keypoints) throws Exception {
        //setup
        int i, j, mx, tm, sc;
        int n_b = params.n_b, a = params.a; //just for readability

        int halfn_b = n_b/2;

        double[] gfilter;
        gfilter = gaussian((double)halfn_b*a, n_b*a);
        
        //compute gradients across series
        double[][] globalGradients = new double[series.length][];
        for(i=0; i<series.length; ++i) {
            globalGradients[i] = new double[series[i].length];

            globalGradients[i][0] = series[i][1] - series[i][0];
            for(j=1; j < series[i].length-1; ++j)
                globalGradients[i][j] = (series[i][j+1] - series[i][j-1]) * 0.5;
            globalGradients[i][j] = series[i][j] - series[i][j-1];
        }

        //compute gradients within each block of each keypoint
        double[][] localGradients = new double[keypoints.size()][];
        for(i = 0; i < localGradients.length; ++i)
            localGradients[i] = new double[a*n_b+1];
        
        for(i=0; i<localGradients.length; ++i) {
            sc = keypoints.get(i).scale;
            tm = keypoints.get(i).time - a*halfn_b;
            mx = keypoints.get(i).time + a*halfn_b; //TODO BUG DEBUG 
            //original code had (- 1), i found that this code gave index out of bound error on line 
            //' localGradients[i][j] = gfilter[j] * globalGradients[sc][tm+j]; ' 
            //with tm+j being the cuplrit
            //authors could not reproduce error in c++ code, unsure of cause
            //effect of change is minimal

            if(tm > 0 && mx < series[0].length) 
                for(j=0; j<=a*n_b; ++j)
                    localGradients[i][j] = gfilter[j] * globalGradients[sc][tm+j];
            else // If near extrema
                for(j=0; j<=a*n_b; ++j)
                    if( (tm+j) < series[0].length && (tm+j) > 0 )
                        localGradients[i][j] = gfilter[j] * globalGradients[sc][tm+j];
            
            //deleting mid element
            double[] temp = new double[localGradients[i].length-1];
            int mid = (int)(localGradients[i].length/2);
            for (j = 0; j < mid; ++j)
                temp[j] = localGradients[i][j];
            for (j = mid+1; j < localGradients[i].length; ++j)
                temp[j-1] = localGradients[i][j];
            localGradients[i] = temp;
        }

        //sum local gradients to form final features
        double[][] features = new double[keypoints.size()][];
        for(i = 0; i < features.length; ++i)
            features[i] = new double[2*n_b];

        for(i=0; i<features.length; ++i)
            for(j=0; j<n_b; ++j)
                for(mx=0; mx<a; ++mx)
                    if(localGradients[i][j*a+mx] < 0)
                        features[i][2*j] -= localGradients[i][j*a+mx];
                    else
                        features[i][2*j+1] += localGradients[i][j*a+mx];

        return features;
    }
    
    public ArrayList<KeyPoint> findDenseKeypoints(GuassianData gdata) {
	int scales = gdata.DoGs.length -1;
	int times = gdata.DoGs[0].length;
        int pointsPerScale = times/params.denseSampleRate;
        
        ArrayList<KeyPoint> keypoints = new ArrayList<>(scales*pointsPerScale);
        
	for(int scale = 1; scale < scales; scale++)
            for(int time = 0; time < times; time+=params.denseSampleRate)
                keypoints.add(new KeyPoint(time, scale));
        
	return keypoints;
    }
    
    public double[] applyGuassian(double[] ts, double sigma) {
	double[] r = new double[ts.length];
        double[] vg = gaussian(sigma);
        
	int i, j, k, m;
	int dec;
	
	dec = (int) ((vg.length + 1) * 0.5);

	for(i=0; i<ts.length; ++i) {
            k = i-dec;
            m = 1;

            for(j=0; j<vg.length; ++j) {
                if(Math.abs(++k) < ts.length)
                    r[i] += (vg[j] * ts[Math.abs(k)]);
                else
                    r[i] += (vg[j] * ts[ts.length-(++m)]);
            }
	}

	return r;
    }

    public double[] gaussian(double sigma) {
	int qs = (int) (4* sigma);
	double[] vg = new double[1 + 2*qs];

	int x, y = -1;
	for(x=-qs; x<=qs; ++x)
            vg[++y] = Math.exp(-1. * x * x / (2.*sigma*sigma) ) / ( Math.sqrt(2. * Math.PI) * sigma);

	return vg;
    }
    
    public double[] gaussian(double sigma, int length) {
	double[] vg;
	if(length % 2 == 1)
            vg = new double[length];
	else
            vg = new double[length+1];

	int x, l = vg.length/2;
	for(x=1; x<=l; ++x) {
            vg[l-x] = Math.exp(-1. * x * x / (2.*sigma*sigma) ) / ( Math.sqrt(2. * Math.PI) * sigma);
            vg[l+x] = Math.exp(-1. * x * x / (2.*sigma*sigma) ) / ( Math.sqrt(2. * Math.PI) * sigma);
	}

	vg[l] = 1. / ( Math.sqrt(2. * Math.PI) * sigma);

        
        double max = vg[0];
	for (int i = 1; i < vg.length; ++i)
            if (vg[i] > max)
                max = vg[i];
	for(int i = 0; i < vg.length; ++i)
		vg[i] /= max;

	return vg;
    }

    public GuassianData findDoGs(double[] ts, double sigma, double k_sc, int n_sc) {
	int size = ts.length;
        
        //for the guassaion filtered series [0] and DoGs [1]
	GuassianData res = new GuassianData();

	res.guassSeries = new double[n_sc+3][];
        res.DoGs = new double[n_sc+2][];
        
	for(int i =0; i<res.DoGs.length; ++i)
            res.DoGs[i] = new double[size];

	int i, j;

	res.guassSeries[0] = applyGuassian(ts, sigma / k_sc);
	res.guassSeries[1] = applyGuassian(ts, sigma);
	
	for(i=0; i<size; ++i)
            res.DoGs[0][i] = res.guassSeries[1][i] - res.guassSeries[0][i];

	// RANG NORMAUX
	for(j=1; j<res.DoGs.length; ++j) {
            res.guassSeries[j+1] = applyGuassian(ts, Math.pow(k_sc, j) * sigma);

            for(i=0; i<size; ++i)
                res.DoGs[j][i] = res.guassSeries[j+1][i] - res.guassSeries[j][i];
	}

	return res;
    }
    

     /**
     * Assumes class index, if present, is last
     * @return data of passed instance in a double array with the class value removed if present
     */
    protected static double[] toArrayNoClass(Instance inst) {
        int length = inst.numAttributes();
        if (inst.classIndex() >= 0)
            --length;
        
        double[] data = new double[length];
        
        for (int i=0, j=0; i < inst.numAttributes(); ++i)
            if (inst.classIndex() != i)
                data[j++] = inst.value(i);
        
        return data;
    }
    
    public void clean() {
        if (clusterData != null)    
            clusterData.clear(); //keeps header info
        if (bagData != null)    
            bagData.clear();
    }
    
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        if(useSVM)
           return classifyInstanceSVM(instnc);
        
        BoTSW_Bag testBag = buildTestBag(instnc);
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        //find dist FROM testBag TO all trainBags
        for (int i = 0; i < bags.length; ++i) {
            double dist = distance(testBag, bags[i], bestDist); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = bags[i].classValue;
            }
        }

        return nn;
    }
    
     /**
     * Used within ensemble as part of a leave-one-out crossvalidation, to skip having to rebuild 
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies 
     * the instance at the index passed while ignoring its own corresponding histogram 
     * 
     * @param test index of instance to classify
     * @return classification
     */
    public double classifyInstance(int test) throws Exception {
        if(useSVM)
           throw new Exception("sped-up loo cv not possible with svm");
        
        BoTSW_Bag testBag = bags[test];
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        //find dist FROM testBag TO all trainBags
        for (int i = 0; i < bags.length; ++i) {
            if (i == test) //skip 'this' one, leave-one-out
                continue;
            
            double dist = distance(testBag, bags[i], bestDist); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = bags[i].classValue;
            }
        }
        
        return nn;
    }
    
    public double distance(BoTSW_Bag instA, BoTSW_Bag instB, double bestDist) {
        double dist = 0.0;
        
        //find dist only from values in instA
        for (int i = 0; i < instA.hist.length; ++i) {
            double valA = instA.hist[i];
            double valB = instB.hist[i];
            dist += (valA-valB)*(valA-valB);
            
            if (dist > bestDist)
                return Double.MAX_VALUE;
        }
        
        return dist;
    }
    
    public double classifyInstanceSVM(Instance instnc) throws Exception {
        double[] dist = distributionForInstance(instnc);
        
        int maxi = 0;
        double max = dist[maxi];
	for (int i = 1; i < dist.length; ++i)
            if (dist[i] > max) {
                max = dist[i];
                maxi = i;
            }
        
        return (double)maxi;
    }
    
    public BoTSW_Bag buildTestBag(Instance instnc) throws Exception {
        double[][] features = extractFeatures(toArrayNoClass(instnc));
        
        //cluster/form histograms
        Instances testFeatures = new Instances(clusterData, features.length);
        double[] hist = new double[params.k];
        for (int i = 0; i < features.length; ++i) {
            testFeatures.add(new DenseInstance(1, features[i]));
            int cluster = kmeans.clusterInstance(testFeatures.get(i));
            ++hist[cluster];
        }
        
        hist = normaliseHistogramSSR(hist);
        hist = normaliseHistograml2(hist);
        
        return new BoTSW_Bag(hist, instnc.classValue());
    }
    
    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        if (useSVM) 
            return distributionForInstanceSVM(instnc);
        else
            throw new UnsupportedOperationException("Not supported yet for non-svm classification."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public double[] distributionForInstanceSVM(Instance instnc) throws Exception {
        BoTSW_Bag testBag = buildTestBag(instnc);
        
        //classify
        Instances testBagData = new Instances(bagData, 1);
        double[] inst = new double[params.k+1];
        for (int j = 0; j < params.k; ++j)
            inst[j] = testBag.hist[j];
        inst[inst.length-1] = testBag.classValue;
        testBagData.add(new DenseInstance(1, inst));
        
        return svm.distributionForInstance(testBagData.get(0));
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args) throws Exception{
        
        //num features for handoutlines as example... == 622307000
        //num doubles with max n_b and a == 9969120000
        //min memory requirement for that == 76058 MB... rofl
//        Params p = new Params(4,4,32);
//        System.out.println("scales: " + p.calcNumScales(2709));
//        long f = (1000*2709*p.calcNumScales(2709));
//        System.out.println("total features: " + f);
//        long MAXdoubles = 8l * 20l * f; //max a = 8, max n_b = 20
//        System.out.println("total num doubles for features: " + MAXdoubles);
//        long MAXmeminbits = MAXdoubles * 64l;
//        long MAXmeminMB = MAXmeminbits/8l/1024l/1024l;
//        System.out.println("total mem in MB: " + MAXmeminMB);
            

        //Minimum working example
        String dataset = "BeetleFly";
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TEST.arff");
        
        Classifier c = new BoTSW();
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BoTSW accuracy on " + dataset + " fold 0 = " + accuracy);
    }
}
