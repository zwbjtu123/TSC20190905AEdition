
package vector_classifiers.weightedvoters;

import timeseriesweka.classifiers.ensembles.voting.NaiveBayesCombiner;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.HESCA;

/**
 * Implemented as separate classifier for explicit comparison, from Kuncheva and Rodríguez (2014)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class HESCA_NaiveBayesCombiner extends HESCA {
    public HESCA_NaiveBayesCombiner() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "HESCA_NaiveBayesCombiner"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new NaiveBayesCombiner();
    }

}
