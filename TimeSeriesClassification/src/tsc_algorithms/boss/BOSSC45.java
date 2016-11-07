/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tsc_algorithms.boss;


import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import tsc_algorithms.BOSS;
import utilities.BitWord;
import utilities.ClassifierTools;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * TEMPORARY WORK AROUND, CONSULT: test cases may contain words not found anywhere 
 * in the train set, and therefore not found in the dictionary (i.e list of attributes
 * in the instance/s). Currently words found in test instances that are not in the train set are 
 * IGNORED. 
 * 
 * 
 * @author xmw13bzu
 */
public class BOSSC45 extends BOSS {

    public boolean buildBags = true;
    public boolean buildForest = true;
    
    public Instances bagInsts;
    public J48 tree = new J48();
    private Attribute classAttribute;
    
    public BOSSC45(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
        super(wordLength, alphabetSize, windowSize, normalise);
    }
    
    public BOSSC45(BOSSC45 boss, int wordLength) {
        super(boss, wordLength);
        
        classAttribute = boss.classAttribute;
    }
    
    public Instances bagsToInstances() {
        //build attribute info, each bag may have different keys etc
        //need to build common vector of all keys found in all bags
        //this is the main source of memory problems, most bags will have many unique
        //keys of value 1 due to noise, thus FULL keyset is much larger than 
        //each individuals bag's keyset
        FastVector<Attribute> attInfo = new FastVector<>();
        Set<String> wordsFound = new HashSet<>();
        for (Bag bag : bags) 
            for (Entry<BitWord,Integer> entry : bag.entrySet()) 
                wordsFound.add(entry.getKey().toString());
        for (String word : wordsFound) 
            attInfo.add(new Attribute(word));
        
        //classvals must not be numeric...            
        attInfo.add(classAttribute);
        
        //atts found, now populate all values
        Instances bagInsts = new Instances("", attInfo, bags.size());
        bagInsts.setClassIndex(attInfo.size()-1);
        
        int i = 0;
        for (Bag bag : bags) {
            //init all values to 0, + class value at the end
            double[] init = new double[attInfo.size()];
            init[init.length-1] = bag.getClassVal();
            
            bagInsts.add(new DenseInstance(1, init));
            for (Entry<BitWord,Integer> entry : bag.entrySet())
                bagInsts.get(i).setValue(bagInsts.attribute(entry.getKey().toString()), entry.getValue());
            
            i++;
        }
        
        return bagInsts;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        classAttribute = data.classAttribute();
        
        if (buildBags)
            super.buildClassifier(data);
            
        if (buildForest)
            buildFullForest();
        //accomodates some speed ups when using cv in ensemble
    }
    
    public void buildFullForest() throws Exception {
        //first pass, building normal bags (ArrayList<Bag>) 
        //then converting to instances for use in rotf
        //TODO potentially build bags straight into instances form, (mem/very slight speed saving) 
        //however the mem savings are made anyway when clean() is called after training
        //and would require a big overhaul
        if (bagInsts == null) 
            bagInsts = bagsToInstances();
        
        if (tree == null) 
            tree = new J48();
        tree.buildClassifier(bagInsts);
    }
    
    public double findCVAcc(int numFolds) throws Exception {
        if (bagInsts == null) 
            bagInsts = bagsToInstances();
        if (tree == null) 
            tree = new J48();
        return ClassifierTools.crossValidationWithStats(tree, bagInsts, numFolds)[0][0];
    }
    
    @Override
    public BOSSC45 buildShortenedBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);
       
        BOSSC45 newBoss = new BOSSC45(this, newWordLength);
        
        //build hists with new word length from SFA words, and copy over the class values of original insts
        for (int i = 0; i < bags.size(); ++i) {
            Bag newBag = createBagFromWords(newWordLength, SFAwords[i]);   
            newBag.setClassVal(bags.get(i).getClassVal());
            newBoss.bags.add(newBag);
        }
        
        return newBoss;
    }
    
    @Override
    public void clean() {
        //null out things that are not needed after training to save memory
        super.clean();
        bags = null;
        tree = null;
        //bagInsts = null; //needed to build full forest if this makes it into the final ensemble
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Bag testBag = BOSSTransform(instance);
        
        //convert bag to instance
        double[] init = new double[bagInsts.numAttributes()];
        init[init.length-1] = testBag.getClassVal();

        //TEMPORARILY create it on the end of the train insts to easily copy over the attribute data.
        bagInsts.add(new DenseInstance(1, init));
        for (Entry<BitWord,Integer> entry : testBag.entrySet()) {
            Attribute att = bagInsts.attribute(entry.getKey().toString());
            if (att != null)
                bagInsts.get(bagInsts.size()-1).setValue(att, entry.getValue());
        }
        
        Instance testInst = bagInsts.remove(bagInsts.size()-1);
        
        return tree.classifyInstance(testInst);
    }
    
    //hacky work around to allow for use of ClassifierTools.
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Bag testBag = BOSSTransform(instance);
        
        //convert bag to instance
        double[] init = new double[bagInsts.numAttributes()];
        init[init.length-1] = testBag.getClassVal();

        //TEMPORARILY create it on the end of the train isnts to easily copy over the attribute data.
        bagInsts.add(new DenseInstance(1, init));
        for (Entry<BitWord,Integer> entry : testBag.entrySet()) {
            Attribute att = bagInsts.attribute(entry.getKey().toString());
            if (att != null)
                bagInsts.get(bagInsts.numInstances()-1).setValue(att, entry.getValue());
        }
        Instance testInst = bagInsts.remove(bagInsts.size()-1);
        
        return tree.distributionForInstance(testInst);
    }
}
