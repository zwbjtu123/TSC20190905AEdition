package weka.filters.timeseries.alternative_shapelet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import utilities.StatisticalUtilities;
import utilities.InstanceTools;
import static utilities.InstanceTools.FromWekaInstances;
import static utilities.StatisticalUtilities.CalculateSigmoid;
import static utilities.StatisticalUtilities.Normalize;
import static utilities.StatisticalUtilities.Normalize2D;
import weka.classifiers.Classifier;
import weka.clusterers.SimpleKMeans;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LearnShapelets implements Classifier{

    // length of a time-series 
    public int seriesLength;
    // length of shapelet
    public int L[];
    public double percentageOfSeriesLength;
    // number of latent patterns
    public int numLatentPatterns;
    // scales of the shapelet length
    public int shapeletLengthScale;
    // number of classes
    public int numClasses;
    // number of segments
    public int J[];
    // shapelets
    double Shapelets[][][];
    // classification weights
    double W[][][];
    double biasW[];

    // the softmax parameter
    public double alpha = -30;
    
    public Instances trainSet;
    public Instance testSet;
    // time series data and the label 
    public double[][] train, classValuePredictions;
    public double[] test;

    // the number of iterations
    public int maxIter = 1000;
    // the learning rate
    public double eta = 0.1;

    public int kMeansIter;

    // the regularization parameters
    public double lambdaW;

    public List<Double> nominalLabels;

    // structures for storing the precomputed terms
    double D_train[][][][];
    double E_train[][][][];
    double M_train[][][];
    double Psi_train[][][];
    double sigY_train[][];
    
    double D_test[][][];
    double E_test[][][];
    double M_test[][];
    double Psi_test[][];
    double sigY_test[];

    Random rand = new Random();

    List<Integer> instanceIdxs;
    List<Integer> rIdxs;

    // constructor
    public LearnShapelets() {
        kMeansIter = 100;
    }

    // initialize the data structures
    public void Initialize() throws Exception {
        
        // avoid K=0 
        if (numLatentPatterns == 0) {
            numLatentPatterns = 1;
        }

        // set the labels to be binary 0 and 1, needed for the logistic loss
        CreateOneVsAllTargets();

        // initialize the shapelets (complete initialization during the clustering)
        Shapelets = new double[shapeletLengthScale][][];
        // initialize the number of shapelets and the length of the shapelets 
        J = new int[shapeletLengthScale];
        L = new int[shapeletLengthScale];
        // set the lengths of shapelets and the number of segments
        // at each scale r
        int totalSegments = 0;
        for (int r = 0; r < shapeletLengthScale; r++) {
            L[r] = (int) ((r + 1) * percentageOfSeriesLength * seriesLength);
            J[r] = seriesLength - L[r];

            totalSegments += train.length * J[r];
        }

        // set the total number of shapelets per scale as a rule of thumb
        // to the logarithm of the total segments
        numLatentPatterns = (int) Math.log(totalSegments);

        // initialize an array of the sizes
        rIdxs = new ArrayList<>();
        for (int r = 0; r < shapeletLengthScale; r++) {
            rIdxs.add(r);
        }

        // initialize shapelets
        InitializeShapeletsKMeans();


        // initialize the terms for pre-computation
        D_train = new double[train.length][shapeletLengthScale][numLatentPatterns][];
        E_train = new double[train.length][shapeletLengthScale][numLatentPatterns][];

        for (int i = 0; i < train.length; i++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                for (int k = 0; k < numLatentPatterns; k++) {
                    D_train[i][r][k] = new double[J[r]];
                    E_train[i][r][k] = new double[J[r]];
                }
            }
        }

        // initialize the placeholders for the precomputed values
        M_train = new double[train.length][shapeletLengthScale][numLatentPatterns];
        Psi_train = new double[train.length][shapeletLengthScale][numLatentPatterns];
        sigY_train = new double[train.length][numClasses];

        // initialize the weights
        W = new double[numClasses][shapeletLengthScale][numLatentPatterns];
        biasW = new double[numClasses];

        for (int c = 0; c < numClasses; c++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                for (int k = 0; k < numLatentPatterns; k++) {
                    W[c][r][k] = 2 * rand.nextDouble() - 1;
                }
            }

            biasW[c] = 2 * rand.nextDouble() - 1;
        }

        // precompute the M, Psi, sigY, used later for setting initial W
        for (int i = 0; i < train.length; i++) {
            PreCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);
        }

        // initialize W by learning the model on the centroid data
        LearnFOnlyW();

        // store all the instances indexes for
        instanceIdxs = new ArrayList<>();
        for (int i = 0; i < train.length; i++) {
            instanceIdxs.add(i);
        }
        // shuffle the order for a better convergence
        Collections.shuffle(instanceIdxs);
    }

    // create one-cs-all targets
    public void CreateOneVsAllTargets() {
        numClasses = nominalLabels.size();

        classValuePredictions = new double[train.length][numClasses];

        // initialize the extended representation  
        for (int i = 0; i < train.length; i++) {
            // firts set everything to zero
            for (int c = 0; c < numClasses; c++) {
                classValuePredictions[i][c] =  0;
            }

            // then set the real label index to 1
            //the label is at the end of the series. not the beginning.
            int indexLabel = nominalLabels.indexOf(train[i][train[i].length-1]);
            classValuePredictions[i][indexLabel] = 1.0;
        }

    }

    // initialize the shapelets from the centroids of the segments
    public void InitializeShapeletsKMeans() throws Exception {
        // a multi-threaded parallel implementation of the clustering
        // on thread for each scale r, i.e. for each set of K shapelets at
        // length L_min*(r+1)
        for (Integer r : rIdxs) {
            double[][] segmentsR = new double[train.length * J[r]][L[r]];

            //construct the segments from the train set.
            for (int i = 0; i < train.length; i++) {
                for (int j = 0; j < J[r]; j++) {
                    for (int l = 0; l < L[r]; l++) {
                        segmentsR[i * J[r] + j][l] = train[i][j + l];
                        
                    }
                }
            }

            // normalize segments
            for (int i = 0; i < train.length; i++) {
                for (int j = 0; j < J[r]; j++) {
                    segmentsR[i * J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i * J[r] + j]);
                }
            }

            
            
            //cluster the shapelets.
            Instances ins = InstanceTools.ToWekaInstances(segmentsR);

            SimpleKMeans skm = new SimpleKMeans();
            skm.setNumClusters(numLatentPatterns);
            skm.setMaxIterations(100);
            skm.setSeed((int) (Math.random() * 1000));
            skm.setInitializeUsingKMeansPlusPlusMethod(true);
            skm.buildClusterer(ins);
            Instances centroidsWeka = skm.getClusterCentroids();

            Shapelets[r] = InstanceTools.FromWekaInstances(centroidsWeka);

            if (Shapelets[r] == null) {
                System.out.println("P not set");
            }
        }
    }

    // predict the label value vartheta_i
    public double Predict_i(double[][] M, int c) {
        double Y_hat_ic = biasW[c];

        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                Y_hat_ic += M[r][k] * W[c][r][k];
            }
        }

        return Y_hat_ic;
    }

    // precompute terms
    public void PreCompute(double[][][] D, double[][][] E, double[][] Psi, double[][] M, double[] sigY, double[] series) {
        
        // precompute terms     
        for (int r = 0; r < shapeletLengthScale; r++) {
            //in most cases Shapelets[r].length == numLatentPatterns, this is not always true.
            for (int k = 0; k < Shapelets[r].length; k++) { 
                for (int j = 0; j < J[r]; j++) {
                    // precompute D
                    D[r][k][j] = 0;
                    double err = 0;
                    
                    for (int l = 0; l < L[r]; l++) {
                        err = series[j + l] - Shapelets[r][k][l];
                        D[r][k][j] += err * err;
                    }

                    D[r][k][j] /= (double) L[r];

                    // precompute E
                    E[r][k][j] = Math.exp(alpha * D[r][k][j]);
                }

                // precompute Psi 
                Psi[r][k] = 0;
                for (int j = 0; j < J[r]; j++) {
                    Psi[r][k] += Math.exp(alpha * D[r][k][j]);
                }

                // precompute M 
                M[r][k] = 0;

                for (int j = 0; j < J[r]; j++) {
                    M[r][k] += D[r][k][j] * E[r][k][j];
                }

                M[r][k] /= Psi[r][k];
            }
        }

        for (int c = 0; c < numClasses; c++) {
            sigY[c] = CalculateSigmoid(Predict_i(M, c));
        }
    }

    // compute the MCR on the test set
    public double trainSetErrorRate() throws Exception {
        int numErrors = 0;

        for (int i=0; i < trainSet.numInstances(); i++) {
            PreCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            Instance inst = trainSet.get(i);
            double label_i = classifyInstance(inst);

            //we've predicted a value which doesn't exist.
            if (inst.classValue() != label_i) {
                numErrors++;
            }
        }

        return (double) numErrors / (double) trainSet.numInstances();
    }

    // compute the accuracy loss of instance i according to the 
    // smooth hinge loss 
    public double AccuracyLoss(double[][] M, double[] Y_b, int c) {
        double Y_hat_ic = Predict_i(M, c);
        double sig_y_ic = CalculateSigmoid(Y_hat_ic);

        return -Y_b[c] * Math.log(sig_y_ic) - (1 - Y_b[c]) * Math.log(1 - sig_y_ic);
    }

    // compute the accuracy loss of the train set
    public double AccuracyLossTrainSet() {
        double accuracyLoss = 0;

        for (int i = 0; i < train.length; i++) {
            PreCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            for (int c = 0; c < numClasses; c++) {
                accuracyLoss += AccuracyLoss(M_train[i], classValuePredictions[i], c);
            }
        }

        return accuracyLoss;
    }

    public void LearnF() {
        // parallel implementation of the learning, one thread per instance
        // up to as much threads as JVM allows
        
        //instanceIdxs is a random reordering of the trin set.
        for (Integer i : instanceIdxs) {
            double regWConst = ((double) 2.0 * lambdaW) / ((double) train.length);

            double tmp2, tmp1, dLdY, dMdS;

            PreCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            for (int c = 0; c < numClasses; c++) {
                dLdY = -(classValuePredictions[i][c] - sigY_train[i][c]);

                for (int r = 0; r < shapeletLengthScale; r++) {
                    //in most cases Shapelets[r].length == numLatentPatterns, this is not always true.
                    for (int k = 0; k < Shapelets[r].length; k++) {
                        W[c][r][k] -= eta * (dLdY * M_train[i][r][k] + regWConst * W[c][r][k]);

                        tmp1 = (2.0 / ((double) L[r] * Psi_train[i][r][k]));

                        for (int l = 0; l < L[r]; l++) {
                            tmp2 = 0;
                            for (int j = 0; j < J[r]; j++) {
                                tmp2 += E_train[i][r][k][j] * (1 + alpha * (D_train[i][r][k][j] - M_train[i][r][k])) * (Shapelets[r][k][l] - train[i][j + l]);
                            }

                            dMdS = tmp1 * tmp2;

                            Shapelets[r][k][l] -= eta * (dLdY * W[c][r][k] * dMdS);

                        }
                    }
                }

                biasW[c] -= eta * dLdY;
            }
        }
    }

    public void LearnFOnlyW() {
        double regWConst = ((double) 2.0 * lambdaW) / ((double) train.length);

        for (int epochs = 0; epochs < maxIter; epochs++) {
            for (int i = 0; i < train.length; i++) {
                for (int c = 0; c < numClasses; c++) {
                    sigY_train[i][c] = CalculateSigmoid(Predict_i(M_train[i], c));

                    for (int r = 0; r < shapeletLengthScale; r++) {
                        for (int k = 0; k < numLatentPatterns; k++) {
                            W[c][r][k] -= eta * (-(classValuePredictions[i][c] - sigY_train[i][c]) * M_train[i][r][k] + regWConst * W[c][r][k]);
                        }
                    }

                    biasW[c] -= eta * (-(classValuePredictions[i][c] - sigY_train[i][c]));
                }
            }
        }

    }

    // optimize the objective function
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        trainSet = data;
        
        seriesLength = trainSet.numAttributes() - 1; //so we don't include the classLabel at the end.

        nominalLabels = ReadNominalTargets(trainSet);
        
        //convert the training set into a 2D Matrix.
        train = FromWekaInstances(trainSet);
        Normalize2D(train, true);
        
        // initialize the data structures
        Initialize();

        List<Double> lossHistory = new ArrayList<>();
        lossHistory.add(Double.MIN_VALUE);

        // apply the stochastic gradient descent in a series of iterations
        for (int iter = 0; iter <= maxIter; iter++) {
            // learn the latent matrices
            LearnF();

            // measure the loss
            if (iter % 500 == 0) {
                double mcrTrain = trainSetErrorRate();

                double lossTrain = AccuracyLossTrainSet();

                lossHistory.add(lossTrain);
                // if divergence is detected start from the beggining 
                // at a lower learning rate
                if (Double.isNaN(lossTrain) || mcrTrain == 1.0) {
                    iter = 0;

                    eta /= 3;

                    lossHistory.clear();

                    Initialize();
                }

                if (lossHistory.size() > 500) {
                    if (lossTrain > lossHistory.get(lossHistory.size() - 2)) {
                        break;
                    }
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
            
        testSet = instance;
        
        test = testSet.toDoubleArray();
        
        test = Normalize(test, true);
        
        // initialize the terms for pre-computation
        D_test = new double[shapeletLengthScale][numLatentPatterns][];
        E_test = new double[shapeletLengthScale][numLatentPatterns][];

        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                D_test[r][k] = new double[J[r]];
                E_test[r][k] = new double[J[r]];
            }
        }

        // initialize the placeholders for the precomputed values
        M_test = new double[shapeletLengthScale][numLatentPatterns];
        Psi_test = new double[shapeletLengthScale][numLatentPatterns];
        sigY_test = new double[numClasses];
        
        PreCompute(D_test, E_test, Psi_test, M_test, sigY_test, test);


        double max_Y_hat_ic = Double.MIN_VALUE;
        int label_i = -1;

        for (int c = 0; c < numClasses; c++) {
            double Y_hat_ic = CalculateSigmoid(Predict_i(M_test, c));

            if (Y_hat_ic > max_Y_hat_ic) {
                max_Y_hat_ic = Y_hat_ic;
                label_i = c;
            }
        }

        return label_i;
    }

    
    public void PrintShapeletsAndWeights() {
        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                System.out.print("Shapelets(" + r + "," + k + ")= [ ");

                for (int l = 0; l < L[r]; l++) {
                    System.out.print(Shapelets[r][k][l] + " ");
                }

                System.out.println("]");
            }
        }

        for (int c = 0; c < numClasses; c++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                System.out.print("W(" + c + "," + r + ")= [ ");

                for (int k = 0; k < numLatentPatterns; k++) {
                    System.out.print(W[c][r][k] + " ");
                }

                System.out.print(biasW[c] + " ");

                System.out.println("]");
            }
        }
    }

    public void PrintProjectedData() {
        int r = 0, c = 0;

        System.out.print("Data= [ ");

        for (int i = 0; i < train.length; i++) {
            PreCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);
            
            System.out.print(classValuePredictions[i][c] + " ");

            for (int k = 0; k < numLatentPatterns; k++) {
                System.out.print(M_train[i][r][k] + " ");
            }

            System.out.println(";");
        }

        System.out.println("];");
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static ArrayList<Double> ReadNominalTargets(Instances instances)
    {
        if (instances.size() <= 0)  return null;
        
        ArrayList<Double> nominalLabels = new ArrayList<>();
        
        for (Instance ins : instances) {
            boolean alreadyAdded = false;

            for (Double nominalLabel : nominalLabels) {
                if (nominalLabel == ins.classValue()) {
                    alreadyAdded = true;
                    break;
                }
            }

            if (!alreadyAdded) {
                nominalLabels.add(ins.classValue());
            }
        }

        Collections.sort(nominalLabels);

        return nominalLabels;
    }
}
