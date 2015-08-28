
package JamesStuff;

import fileIO.InFile;
import fileIO.OutFile;

/**
 * UNTESTED as of 24/08
 * 
 * Code to parse the ifr alcohol data into arff formats.
 * 
 * Currently, it simply does the conversion based on the criteria/class labels provided ('Experiment') into a single dataset. i.e
 * does not split data into train and test datasets, only a single dump file
 * 
 * todo could make experiment etc into full class if wanted, could be useful elsewhere as a general raw data -> arff file parser
 * i.e -user provides 'file reader' by implementing some interface to read the values/attributes from whatever file format the data is in 
 *     -provide list of 'experiments' i.e ways of splitting data into files and corresponding class values
 *     -output set of named arff files (one for each experiment, which can then be split into train/test as needed etc)
 * 
 * todo different experiments for this dataset(i.e ways of grouping the data into files/different class values)? 
 *          -binary class problem of normal vs odd bottles regardless of contents
 *          -classifying based on individual bottle regardless of contents
 *          -binary classification of 'legal' vs 'illegal' regardless of specific concentration
 * 
 * 
 * @author James
 */
public class AlcoholDataParser {
    
    public static final String masterPath = "C:\\\\Temp\\\\ifr\\\\AlchoholData\\\\";
    public static final String inputPath = masterPath + "All\\\\";
    public static final String outputPath = masterPath + "PROCESSED\\\\";
    
    public static final String[] bottles = {
        //"Normal" bottles
        "aberfeldy",
        "aberlour", //difficult, seams/stickers in way
        "amrut",
        "ancnoc",
        "armorik",
        "arran10",
        "arran14",
        "asyla",
        "benromach",
        "bladnoch", //difficult, seams/stickers in way
        "blairathol",
        "exhibition",
        "glencadam",
        "glendeveron",
        "glenfarclas",
        "glengoyne",
        "glenlivet15",
        "glenmorangie",
        "glenmoray",
        "glenscotia",
        "oakcross",
        "organic",
        "peatmonster",
        "scapa", //difficult, seams/stickers in way
        "smokehead",
        "speyburn",
        "spicetree",
        "taliskerenglishwhisky9",
            
        //"Odd" shaped bottles
        "allmalt",
        "auchentoshan",
        "balblair",
        "bernheim",
        "cardhu",
        "dutchsinglemalt",
        "elijahcraig",
        "englishwhisky15",
        "glenfiddich",
        "greatkingst",
        "highlandpark",
        "mackmyra", 
        "nikka",
        
        //Green glass bottles
        "finlaggan",
        "glenlivet12",
        "laphroaig"
    };
    
    public enum Experiment {
        ETHANOL("Ethanol4Class"), //Classify based on ethanol content regardless of bottle
        METHANOL("Methanol5Class"), //Classify based on methanol content regardless of bottle
        BOTTLE("BottleClassification"); //Classify based on bottle regardless of contents
        
        private final String experimentName;
        private Experiment (String en) {
            experimentName=en;
        }
        public String getName() { return experimentName; }
    }
    
    //again export to full experiment class  if it's made eventually
    //different experiments may want to use different spectral bands
    public static final double minWaveLength = 226.0;
    public static final double maxWaveLength = 1101.5;
    public static final double waveLengthIncrement = 0.5;
    
    public static final String[/*Experiment*/][] expConcentrations = {  
        { "E35", "E38", "E40", "E45" },    //Ethanol
        { "E40", "M05", "M1", "M2", "M5" }, //Methanol
        { "E35", "E38", "E40", "E45", "M05", "M1", "M2", "M5" } //Bottle
    };

    public static final String[/*Experiment*/][] expClassValues = {
        { "35%", "38%", "40%", "45%" },   //Ethanol
        { "0%", "0.5%", "1%", "2%", "5%"}, //Methanol
        bottles
    };

    public static void main(String[] args) {
        for (Experiment exp : Experiment.values()) {
            OutFile out = new OutFile(outputPath + exp.getName() + ".arff");
            
            writeMetaData(out, exp, expClassValues[exp.ordinal()]);

            for (int con = 0; con < expConcentrations[exp.ordinal()].length; con++)
                for (int bottle = 0; bottle < bottles.length; ++bottle)
                    for (int batch = 1; batch <= 3; batch++)
                        for (int rep = 1; rep <= 3; rep++)
                            SSMFileToARFF(out, exp, con, bottle, batch, rep);

            out.closeFile();
        }
    }
    
    public static void writeMetaData(OutFile out, Experiment exp, String[] classValues) {
        out.writeLine("@Relation " + exp.getName() + "\r");
        out.writeLine("\r");
        
        if (exp != Experiment.BOTTLE)
            writeBottleAttribute(out);
        writeWaveLengthAttributes(out);
        writeClassAttribute(out, classValues);
        
        out.writeLine("\r");
        out.writeLine("@Data\r");
    }
    
    public static void writeBottleAttribute(OutFile out) {
        out.writeString("@attribute bottleName {" + bottles[0]);
        for (String bottle : bottles) 
            out.writeString("," + bottle);
        out.writeLine("}\r");
    }
    
    public static void writeWaveLengthAttributes(OutFile out) {
        SSMFileReader testFile = new SSMFileReader(masterPath + "off_nothing.SSM");
        for (double i = minWaveLength; i < maxWaveLength; i+=waveLengthIncrement) 
            out.writeLine("@attribute wavelength_" + i + "\r");
        testFile.closeFile();
    }
    
    public static void writeClassAttribute(OutFile out, String[] classValues) {
        out.writeString("@attribute classValues {");
        for (String val : classValues) 
            out.writeString(val + ",");
        out.writeLine("}\r");
    }
    
    public static void SSMFileToARFF(OutFile out, Experiment exp, int con, int bottle, int batch, int rep) {
        String path = inputPath + expConcentrations[exp.ordinal()][con] + "_" + bottles[bottle] + "_" + batch + "_" + rep + ".SSM";
        SSMFileReader reading;

        try {
            reading = new SSMFileReader(path);
        } catch (Exception e) {
            System.out.println("Error opening file: " + path + "\n" + e);
            return; //sample may not exist (may have missed / overwritten it etc), not necessarily incorrect logic 
        }

        try {
            if (exp != Experiment.BOTTLE)
                out.writeString(bottle + ",");

            //update at some point, currently assuming just using all wavelengths
            //eventually will likely want to use particular spectral bands, so write out 
            //only relevant values (skip preceding ones)
            for (double i = minWaveLength; i < maxWaveLength; i+=waveLengthIncrement)
                out.writeString(reading.nextValue() + ",");

            if (exp == Experiment.BOTTLE)
                out.writeInt(bottle);
            else 
                out.writeInt(con);
            
            out.writeLine("\r");
        } catch (Exception e) {
            System.out.println("Error writing to file: " + path);
            throw e;
        }

        reading.closeFile();
    }
    
    public static class SSMFileReader {
        String file;
        String metaData;
        double[] data;
        
        InFile in;
        
        public SSMFileReader(String file) {
            this.file = file;
            
            //add error checking blabla
            in = new InFile(file);
            
            in.readLine(); //file path/name again
            metaData = in.readLine(); //actual meta data, expand this later if needed 
            //(could use to check for irregularity/misreadings, as far as i know all SHOULD be identical aside from 'val')
        }
        
        public double nextWavelength() {
            double[] d = nextReading();
            return d[0];
        }
        
        public double nextValue() {
            double[] d = nextReading();
            return d[1];
        }
        
        public double[] nextReading() {
            double r[] = new double[2];
            r[0] = in.readDouble();
            r[1] = Double.valueOf(in.readLine().trim());//scientific notation
            return r;
        }

        //turns out .isEmpty() doesnt work
//        public boolean eof() {
//            return in.isEmpty();
//        }
//        
        public void closeFile() {
            in.closeFile();
        }
    }
    
}
