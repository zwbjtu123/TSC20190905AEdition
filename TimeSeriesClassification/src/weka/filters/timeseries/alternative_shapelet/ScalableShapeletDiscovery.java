package weka.filters.timeseries.alternative_shapelet;

//import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static utilities.InstanceTools.FromWekaInstances;
import static utilities.StatisticalUtilities.Normalize2D;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class ScalableShapeletDiscovery implements Classifier{

    // series data and labels
    public double[][] trainSeriesData;
    public double[] trainSeriesLabels, testSeriesData;
    public double testSeriesLabel;

    public int numTrainInstances;
    int seriesLength;

    // the length of the shapelet we are searching for
    int[] shapeletLengths;

    // the epsilon parameter to prune candidates being epsilon close to a 
    // rejected or accepted shapelet
    public double epsilon;

    // the percentile for the distribution of distances between pairs of segments
    // is used as epsilon
    public int percentile;

    // list of accepted and rejected words
    List<double[]> acceptedList = new ArrayList<>();
    List<double[]> rejectedList = new ArrayList<>();

    public String trainSetPath, testSetPath;
    // the paa ratio, i.e. 0.25 reduces the length of series by 1/4
    public double paaRatio;

    // the histogram contains lists of frequency columns
    List<double[]> distancesShapelets = new ArrayList<>();
    // the current classification error of the frequencies histogram
    double currentTrainError = Double.MAX_VALUE;

    double[][] seriesDistancesMatrix;

    // logs on the number of acceptances and rejections
    public int numAcceptedShapelets, numRejectedShapelets, numRefusedShapelets;

    public long trainTime, testTime;

    public boolean normalizeData;

    // random number generator. TODO: need a seed for this...
    Random rand = new Random();

    public ScalableShapeletDiscovery() {
        normalizeData = false;
    }

    @Override
    public void buildClassifier(Instances train) {
        
        trainSeriesData = FromWekaInstances(train);
        Normalize2D(trainSeriesData, true);
        
        //TODO: SAX this data baased on the PAA ratio.

        numAcceptedShapelets = numRejectedShapelets = numRefusedShapelets = 0;
        // num train instances
        numTrainInstances = train.numInstances();
        
        //Aaron: Last attribtute is class value.
        seriesLength = train.numAttributes() - 1;
        
        //setup the training labels.
        trainSeriesLabels = new double[numTrainInstances];
        for(int i=0; i < numTrainInstances; i++)
            trainSeriesLabels[i] = train.get(i).classValue();
        

        // check 20%,40%, 60% shapelet lengths
        shapeletLengths = new int[3];
        shapeletLengths[0] = (int) (0.20 * seriesLength);
        shapeletLengths[1] = (int) (0.40 * seriesLength);
        shapeletLengths[2] = (int) (0.60 * seriesLength);

        epsilon = EstimateEpsilon();

        //System.out.println(epsilon); 
        // set distances matrix to 0.0
        seriesDistancesMatrix = new double[numTrainInstances][];
        for (int i = 0; i < numTrainInstances; i++) {
            //Aaron. I changed the array allocation because it was wastfeul.
            seriesDistancesMatrix[i] = new double[numTrainInstances - i + 1];
            for (int j = i + 1; j < numTrainInstances; j++) {
                seriesDistancesMatrix[i][j] = 0.0;
            }
        }

        // evaluate all the words of all series
        int numTotalCandidates = numTrainInstances * seriesLength * shapeletLengths.length;

        //Logging.println("Candidate shapelets:");
        for (int candidateIdx = 0; candidateIdx < numTotalCandidates; candidateIdx++) {
            // select a random series
            int i = rand.nextInt(numTrainInstances);
            // select a random shapelet length
            int shapeletLength = shapeletLengths[rand.nextInt(shapeletLengths.length)];
            // select a random segment of the i-th series
            int j = rand.nextInt(seriesLength - shapeletLength + 1);

            double[] candidateShapelet = new double[shapeletLength];
            for (int k = 0; k < shapeletLength; k++) {
                candidateShapelet[k] = trainSeriesData[i][j + k];
            }

            EvaluateShapelet(candidateShapelet);
        }
    }

    // consider a word whether it is part of the accept list of reject list
    // and if not whether it helps reduce the classification error
    public void EvaluateShapelet(double[] candidate) {

        // if the lists are both empty or the candidate is previously not been considered 
        // then give it a chance
        if (!FoundInList(candidate, acceptedList) && !FoundInList(candidate, rejectedList)) {
            // compute the soft frequencies of the word in the series data
            double[] distancesCandidate = ComputeDistances(candidate);

            // refresh distances
            AddCandidateDistancesToDistancesMatrix(distancesCandidate);
            // compute error
            double newTrainError = ComputeTrainError();

            if (newTrainError < currentTrainError) {
                // accept the word, which improves the error
                acceptedList.add(candidate);

                // add the distances of the shapelet to a list
                // will be used for testing
                distancesShapelets.add(distancesCandidate);

                // set the new error as the current one
                currentTrainError = newTrainError;

                // increase the counter of accepted words
                numAcceptedShapelets++;
            } else {
                // the word doesn't improve the error, therefore is rejected
                rejectedList.add(candidate);

                // finally remove the distances from the distance matrix
                RemoveCandidateDistancesToDistancesMatrix(distancesCandidate);

                // increase the counter of rejected words
                numRejectedShapelets++;
            }

        } else // word already was accepted and/or rejected before
        {
            numRefusedShapelets++;
        }

    }

    // compute the minimum distance of a candidate to the training series
    private double[] ComputeDistances(double[] candidate) {
        double[] distancesCandidate = new double[numTrainInstances];

        double diff, distanceToSegment, minDistanceSoFar;

        int shapeletLength = candidate.length;

        for (int i = 0; i < numTrainInstances; i++) {
            minDistanceSoFar = Double.MAX_VALUE;

            for (int j = 0; j < seriesLength - shapeletLength + 1; j++) {
                distanceToSegment = 0;

                for (int k = 0; k < shapeletLength; k++) {
                    diff = candidate[k] - trainSeriesData[i][j + k];
                    distanceToSegment += diff * diff;

                    // if the distance of the candidate to this segment is more than the best so far
                    // at point k, skip the remaining points
                    if (distanceToSegment > minDistanceSoFar) {
                        break;
                    }
                }

                if (distanceToSegment < minDistanceSoFar) {
                    minDistanceSoFar = distanceToSegment;
                }
            }

            distancesCandidate[i] = minDistanceSoFar;
        }

        return distancesCandidate;
    }

    // compute the error of the training instances
    // from the distances matrix
    public double ComputeTrainError() {
        int numMissClassifications = 0;
        double realLabel;
        double nearestLabel;
        double nearestDistance;

        // for every test instance 
        for (int i = 0; i < numTrainInstances; i++) {
            realLabel = trainSeriesLabels[i];

            nearestLabel = -1;
            nearestDistance = Double.MAX_VALUE;

            // iterate through training instances and find the closest neighbours
            for (int j = 0; j < numTrainInstances; j++) {
                // avoid itself as a neighbor
                if (i == j) {
                    continue;
                }

                double distance = (i < j ? seriesDistancesMatrix[i][j] : seriesDistancesMatrix[j][i]);

                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestLabel = trainSeriesLabels[j];
                }
            }

            if (realLabel != nearestLabel) {
                numMissClassifications += 1.0;
            }
        }

        // return the error rate
        return (double) numMissClassifications / (double) numTrainInstances;
    }

    public void AddCandidateDistancesToDistancesMatrix(double[] candidateDistances) {
        double diff;

        for (int i = 0; i < numTrainInstances; i++) {
            for (int j = i + 1; j < numTrainInstances; j++) {
                diff = candidateDistances[i] - candidateDistances[j];
                seriesDistancesMatrix[i][j] += diff * diff;
            }
        }
    }

    public void RemoveCandidateDistancesToDistancesMatrix(double[] candidateDistances) {
        double diff;

        for (int i = 0; i < numTrainInstances; i++) {
            for (int j = i + 1; j < numTrainInstances; j++) {
                diff = candidateDistances[i] - candidateDistances[j];
                seriesDistancesMatrix[i][j] -= diff * diff;
            }
        }
    }


    @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        testSeriesData = instance.toDoubleArray();
        
        int shapeletLength;

        double minDistanceSoFar, distanceToSegment, diff,  nearestLabel = 0, nearestDistance = Double.MAX_VALUE;

        int numShapelets = distancesShapelets.size();
        
        double[] distTestInstanceToShapelets = new double[numShapelets];

        // compute the distances of the test instance to the shapelets
        for (int shapeletIndex = 0; shapeletIndex < numShapelets; shapeletIndex++) {
            minDistanceSoFar = Double.MAX_VALUE;
            // read the shapelet length
            shapeletLength = acceptedList.get(shapeletIndex).length;

            for (int j = 0; j < seriesLength - shapeletLength + 1; j++) {
                distanceToSegment = 0;

                for (int k = 0; k < shapeletLength; k++) {
                    diff = acceptedList.get(shapeletIndex)[k] - testSeriesData[j + k];
                    distanceToSegment += diff * diff;

                    // if the distance of the candidate to this segment is more than the best so far
                    // at point k, skip the remaining points
                    if (distanceToSegment > minDistanceSoFar) {
                        break;
                    }
                }

                if (distanceToSegment < minDistanceSoFar) {
                    minDistanceSoFar = distanceToSegment;
                }
            }

            distTestInstanceToShapelets[shapeletIndex] = minDistanceSoFar;
        }

        // iterate through training instances and find the closest neighbours
        for (int j = 0; j < numTrainInstances; j++) {
            double distance = 0;

            for (int k = 0; k < numShapelets; k++) {
                double error = distTestInstanceToShapelets[k] - distancesShapelets.get(k)[j];
                distance += error * error;
            }

            // if there are less then required candidates then add it as a candidate directly
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestLabel = trainSeriesLabels[j];
            }
        }
        
        return nearestLabel;
    }
    

    // is a candidate found in the accepted list of rejected list 
    public boolean FoundInList(double[] candidate, List<double[]> list) {
        double diff, distance;
        int shapeletLength = candidate.length;

        for (double[] shapelet : list) {
            // avoid comparing against shapelets of other lengths
            if (shapelet.length != candidate.length) {
                continue;
            }

            distance = 0;
            for (int k = 0; k < shapeletLength; k++) {
                diff = candidate[k] - shapelet[k];
                distance += diff * diff;

                // if the distance so far exceeds epsilon then stop
                if (distance / shapeletLength > epsilon) {
                    break;
                }

            }

            if (distance / shapeletLength < epsilon) {
                return true;
            }
        }

        return false;
    }

    // estimate the pruning distance
    //TODO: FIX!!
    public double EstimateEpsilon() {
        // return 0 epsilon if no pruning is requested, i.e. percentile=0
        if (percentile == 0) {
            return 0;
        }

        int numPairs = numTrainInstances * seriesLength;

        double[] distances = new double[numPairs];

        int seriesIndex1, pointIndex1, seriesIndex2, pointIndex2, shapeletLength;
        double pairDistance, diff;

        //DescriptiveStatistics stat = new DescriptiveStatistics();

        for (int i = 0; i < numPairs; i++) {
            shapeletLength = shapeletLengths[rand.nextInt(shapeletLengths.length)];

            seriesIndex1 = rand.nextInt(numTrainInstances);
            pointIndex1 = rand.nextInt(seriesLength - shapeletLength + 1);

            seriesIndex2 = rand.nextInt(numTrainInstances);
            pointIndex2 = rand.nextInt(seriesLength - shapeletLength + 1);

            pairDistance = 0;
            for (int k = 0; k < shapeletLength; k++) {
                diff = trainSeriesData[seriesIndex1][pointIndex1 + k] - trainSeriesData[seriesIndex2][pointIndex2 + k];
                pairDistance += diff * diff;
            }

            distances[i] = pairDistance / (double) shapeletLength;

            //stat.addValue(distances[i]);

        }

        return 0;//stat.getPercentile(percentile);
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
