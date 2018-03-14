
package vector_classifiers.weightedvoters;

import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import vector_classifiers.HESCA;

/**
 * Implemented as separate classifier for explicit comparison, from Kuncheva and Rodríguez (2014)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class HESCA_WeightedMajorityVote extends HESCA {
    public HESCA_WeightedMajorityVote() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "HESCA_WeightedMajorityVote"; 
        weightingScheme = new TrainAcc();
        votingScheme = new MajorityVote();
    } 
}
