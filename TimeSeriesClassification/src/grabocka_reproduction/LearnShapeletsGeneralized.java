package grabocka_reproduction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import utilities.StatisticalUtilities;
import utilities.InstanceTools;
import static utilities.StatisticalUtilities.CalculateSigmoid;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class LearnShapeletsGeneralized {

    // number of training and testing instances
    public int ITrain, ITest;
    // length of a time-series 
    public int Q;
    // length of shapelet
    public int L[];
    public int L_min;
    // number of latent patterns
    public int K;
    // scales of the shapelet length
    public int R;
    // number of classes
    public int C;
    // number of segments
    public int J[];
    // shapelets
    double Shapelets[][][];
    // classification weights
    double W[][][];
    double biasW[];

    // the softmax parameter
    public double alpha;

    // time series data and the label 
    public double[][] T, Y, Y_b;

    // the number of iterations
    public int maxIter;
    // the learning rate
    public double eta;

    public int kMeansIter;

    // the regularization parameters
    public double lambdaW;

    public List<Double> nominalLabels;

    // structures for storing the precomputed terms
    double D[][][][];
    double E[][][][];
    double M[][][];
    double Psi[][][];
    double sigY[][];

    Random rand = new Random();

    List<Integer> instanceIdxs;
    List<Integer> rIdxs;

    // constructor
    public LearnShapeletsGeneralized() {
        kMeansIter = 100;
    }

    // initialize the data structures
    public void Initialize() {
        // avoid K=0 
        if (K == 0) {
            K = 1;
        }

        // set the labels to be binary 0 and 1, needed for the logistic loss
        CreateOneVsAllTargets();

        // initialize the shapelets (complete initialization during the clustering)
        Shapelets = new double[R][][];
        // initialize the number of shapelets and the length of the shapelets 
        J = new int[R];
        L = new int[R];
        // set the lengths of shapelets and the number of segments
        // at each scale r
        int totalSegments = 0;
        for (int r = 0; r < R; r++) {
            L[r] = (r + 1) * L_min;
            J[r] = Q - L[r];

            totalSegments += ITrain * J[r];
        }

        // set the total number of shapelets per scale as a rule of thumb
        // to the logarithm of the total segments
        K = (int) Math.log(totalSegments);

        // initialize an array of the sizes
        rIdxs = new ArrayList<>();
        for (int r = 0; r < R; r++) {
            rIdxs.add(r);
        }

        try {
            // initialize shapelets
            InitializeShapeletsKMeans();
        } catch (Exception ex) {
            System.out.println("Exception initialising KMeans: " + ex);
        }

        // initialize the terms for pre-computation
        D = new double[ITrain + ITest][R][K][];
        E = new double[ITrain + ITest][R][K][];

        for (int i = 0; i < ITrain + ITest; i++) {
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < K; k++) {
                    D[i][r][k] = new double[J[r]];
                    E[i][r][k] = new double[J[r]];
                }
            }
        }

        // initialize the placeholders for the precomputed values
        M = new double[ITrain + ITest][R][K];
        Psi = new double[ITrain + ITest][R][K];
        sigY = new double[ITrain + ITest][C];

        // initialize the weights
        W = new double[C][R][K];
        biasW = new double[C];

        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < K; k++) {
                    W[c][r][k] = 2 * rand.nextDouble() - 1;
                }
            }

            biasW[c] = 2 * rand.nextDouble() - 1;
        }

        // precompute the M, Psi, sigY, used later for setting initial W
        for (int i = 0; i < ITrain + ITest; i++) {
            PreCompute(i);
        }

        // initialize W by learning the model on the centroid data
        LearnFOnlyW();

        // store all the instances indexes for
        instanceIdxs = new ArrayList<>();
        for (int i = 0; i < ITrain; i++) {
            instanceIdxs.add(i);
        }
        // shuffle the order for a better convergence
        Collections.shuffle(instanceIdxs);

    }

    // create one-cs-all targets
    public void CreateOneVsAllTargets() {
        C = nominalLabels.size();

        Y_b = new double[ITrain + ITest][C];

        // initialize the extended representation  
        for (int i = 0; i < ITrain + ITest; i++) {
            // firts set everything to zero
            for (int c = 0; c < C; c++) {
                Y_b[i][c] =  0;
            }

            // then set the real label index to 1
            //the label is at the end of the series. not the beginning.
            int indexLabel = nominalLabels.indexOf(Y[i][Y[i].length-1]);
            Y_b[i][indexLabel] = 1.0;
        }

    }

    // initialize the shapelets from the centroids of the segments
    public void InitializeShapeletsKMeans() throws Exception {
        // a multi-threaded parallel implementation of the clustering
        // on thread for each scale r, i.e. for each set of K shapelets at
        // length L_min*(r+1)
        for (Integer r : rIdxs) {
            double[][] segmentsR = new double[(ITrain) * J[r]][L[r]];

            //construct the segments from the train set.
            for (int i = 0; i < (ITrain); i++) {
                for (int j = 0; j < J[r]; j++) {
                    for (int l = 0; l < L[r]; l++) {
                        segmentsR[i * J[r] + j][l] = T[i][j + l];
                    }
                }
            }

            // normalize segments
            for (int i = 0; i < (ITrain); i++) {
                for (int j = 0; j < J[r]; j++) {
                    segmentsR[i * J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i * J[r] + j]);
                }
            }

            //cluster the shapelets.
            Instances ins = InstanceTools.ToWekaInstances(segmentsR);

            SimpleKMeans skm = new SimpleKMeans();
            skm.setNumClusters(K);
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
    public double Predict_i(int i, int c) {
        double Y_hat_ic = biasW[c];

        for (int r = 0; r < R; r++) {
            for (int k = 0; k < K; k++) {
                Y_hat_ic += M[i][r][k] * W[c][r][k];
            }
        }

        return Y_hat_ic;
    }

    // precompute terms
    public void PreCompute(int i) {
        // precompute terms
        for (int r = 0; r < R; r++) {
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < J[r]; j++) {
                    // precompute D
                    D[i][r][k][j] = 0;
                    double err = 0;

                    for (int l = 0; l < L[r]; l++) {
                        err = T[i][j + l] - Shapelets[r][k][l];
                        D[i][r][k][j] += err * err;
                    }

                    D[i][r][k][j] /= (double) L[r];

                    // precompute E
                    E[i][r][k][j] = Math.exp(alpha * D[i][r][k][j]);
                }

                // precompute Psi 
                Psi[i][r][k] = 0;
                for (int j = 0; j < J[r]; j++) {
                    Psi[i][r][k] += Math.exp(alpha * D[i][r][k][j]);
                }

                // precompute M 
                M[i][r][k] = 0;

                for (int j = 0; j < J[r]; j++) {
                    M[i][r][k] += D[i][r][k][j] * E[i][r][k][j];
                }

                M[i][r][k] /= Psi[i][r][k];
            }
        }

        for (int c = 0; c < C; c++) {
            sigY[i][c] = CalculateSigmoid(Predict_i(i, c));
        }
    }

    // compute the MCR on the test set
    public double GetMCRTrainSet() {
        int numErrors = 0;

        for (int i = 0; i < ITrain; i++) {
            PreCompute(i);

            double max_Y_hat_ic = Double.MIN_VALUE;
            int label_i = -1;

            for (int c = 0; c < C; c++) {
                double Y_hat_ic = CalculateSigmoid(Predict_i(i, c));

                if (Y_hat_ic > max_Y_hat_ic) {
                    max_Y_hat_ic = Y_hat_ic;
                    label_i = c;
                }
            }

            //we've predicted a value which doesn't exist.
            if (nominalLabels.indexOf(Y[i]) != label_i) {
                numErrors++;
            }
        }

        return (double) numErrors / (double) ITrain;
    }

    // compute the MCR on the test set
    private double GetMCRTestSet() {
        int numErrors = 0;

        for (int i = ITrain; i < ITrain + ITest; i++) {
            PreCompute(i);

            double max_Y_hat_ic = Double.MIN_VALUE;
            int label_i = -1;

            for (int c = 0; c < C; c++) {
                double Y_hat_ic = CalculateSigmoid(Predict_i(i, c));

                if (Y_hat_ic > max_Y_hat_ic) {
                    max_Y_hat_ic = Y_hat_ic;
                    label_i = c;
                }
            }

            if (nominalLabels.indexOf(Y[i]) != label_i) {
                numErrors++;
            }
        }

        return (double) numErrors / (double) ITest;
    }

    // compute the accuracy loss of instance i according to the 
    // smooth hinge loss 
    public double AccuracyLoss(int i, int c) {
        double Y_hat_ic = Predict_i(i, c);
        double sig_y_ic = CalculateSigmoid(Y_hat_ic);

        return -Y_b[i][c] * Math.log(sig_y_ic) - (1 - Y_b[i][c]) * Math.log(1 - sig_y_ic);
    }

    // compute the accuracy loss of the train set
    public double AccuracyLossTrainSet() {
        double accuracyLoss = 0;

        for (int i = 0; i < ITrain; i++) {
            PreCompute(i);

            for (int c = 0; c < C; c++) {
                accuracyLoss += AccuracyLoss(i, c);
            }
        }

        return accuracyLoss;
    }

    // compute the accuracy loss of the train set
    public double AccuracyLossTestSet() {
        double accuracyLoss = 0;

        for (int i = ITrain; i < ITrain + ITest; i++) {
            PreCompute(i);

            for (int c = 0; c < C; c++) {
                accuracyLoss += AccuracyLoss(i, c);
            }
        }
        return accuracyLoss;
    }

    public void LearnF() {
        // parallel implementation of the learning, one thread per instance
        // up to as much threads as JVM allows
        for (Integer i : instanceIdxs) {
            double regWConst = ((double) 2.0 * lambdaW) / ((double) ITrain);

            double tmp2 = 0, tmp1 = 0, dLdY = 0, dMdS = 0;

            PreCompute(i);

            for (int c = 0; c < C; c++) {
                dLdY = -(Y_b[i][c] - sigY[i][c]);

                for (int r = 0; r < R; r++) {
                    for (int k = 0; k < K; k++) {
                        W[c][r][k] -= eta * (dLdY * M[i][r][k] + regWConst * W[c][r][k]);

                        tmp1 = (2.0 / ((double) L[r] * Psi[i][r][k]));

                        for (int l = 0; l < L[r]; l++) {
                            tmp2 = 0;
                            for (int j = 0; j < J[r]; j++) {
                                tmp2 += E[i][r][k][j] * (1 + alpha * (D[i][r][k][j] - M[i][r][k])) * (Shapelets[r][k][l] - T[i][j + l]);
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
        double regWConst = ((double) 2.0 * lambdaW) / ((double) ITrain);

        for (int epochs = 0; epochs < maxIter; epochs++) {
            for (int i = 0; i < ITrain; i++) {
                for (int c = 0; c < C; c++) {
                    sigY[i][c] = CalculateSigmoid(Predict_i(i, c));

                    for (int r = 0; r < R; r++) {
                        for (int k = 0; k < K; k++) {
                            W[c][r][k] -= eta * (-(Y_b[i][c] - sigY[i][c]) * M[i][r][k] + regWConst * W[c][r][k]);
                        }
                    }

                    biasW[c] -= eta * (-(Y_b[i][c] - sigY[i][c]));
                }
            }
        }

    }

    // optimize the objective function
    public double Learn() {
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
                double mcrTrain = GetMCRTrainSet();

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

        return GetMCRTestSet();
    }

    public void PrintShapeletsAndWeights() {
        for (int r = 0; r < R; r++) {
            for (int k = 0; k < K; k++) {
                System.out.print("Shapelets(" + r + "," + k + ")= [ ");

                for (int l = 0; l < L[r]; l++) {
                    System.out.print(Shapelets[r][k][l] + " ");
                }

                System.out.println("]");
            }
        }

        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                System.out.print("W(" + c + "," + r + ")= [ ");

                for (int k = 0; k < K; k++) {
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

        for (int i = 0; i < ITrain; i++) {
            PreCompute(i);

            System.out.print(Y_b[i][c] + " ");

            for (int k = 0; k < K; k++) {
                System.out.print(M[r][k] + " ");
            }

            System.out.println(";");
        }

        System.out.println("];");
    }
}
