/*
AJB Oct 2016

Model to simulate data where dictionary approach should be optimal.

A single shapelet is common to all series.  The discriminatory feature is the 
number of times it appears in a series. 


*/
package statistics.simulators;
import fileIO.OutFile;
import java.util.*;
import java.io.*;
import statistics.distributions.NormalDistribution;

public class DictionaryModel extends Model {
    
    public enum ShapeType {TRIANGLE,HEADSHOULDERS,SINE, STEP, SPIKE};
    private static int DEFAULTNUMSHAPELETS=5;
    private static int DEFAULTSERIESLENGTH=1000;
    public static int DEFAULTSHAPELETLENGTH=29;

    
    protected Shape shape1;
    protected Shape shape2;
    protected int numShape1=DEFAULTNUMSHAPELETS;
    protected int[] shape1Locations=new int[numShape1];
    protected int numShape2=DEFAULTNUMSHAPELETS;
    protected int[] shape2Locations=new int[numShape2];
    
    protected int totalNumShapes=numShape1+numShape2;
    protected int seriesLength=DEFAULTSERIESLENGTH; 
    protected int shapeletLength=DEFAULTSHAPELETLENGTH;

    //Default Constructor, max start should be at least 29 less than the length
    // of the series if using the default shapelet length of 29
    public DictionaryModel()
    {
        this(new double[]{DEFAULTSERIESLENGTH,DEFAULTNUMSHAPELETS,DEFAULTNUMSHAPELETS,DEFAULTSHAPELETLENGTH});
    }
    public DictionaryModel(double[] param)
    {
        super();
        setDefaults();
//PARAMETER LIST: seriesLength,  numShape1, numShape2, shapeletLength
        if(param!=null){
            switch(param.length){
                default: 
                case 4:             shapeletLength=(int)param[3];
                case 3:             numShape2=(int)param[2];
                case 2:             numShape1=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
        totalNumShapes=numShape1+numShape2;
        shape1Locations=new int[numShape1];
        shape2Locations=new int[numShape2];
        shape1=new Shape();
        shape1.type=ShapeType.SINE;
        shape2=new Shape();
        shape2.type=ShapeType.HEADSHOULDERS;
//Enforce non-overlapping, only occurs if there is not enough room for 
//all the shapes. Locations split randomly between the two classes        
        while(!setNonOverlappingLocations()){
            totalNumShapes--;
            if(numShape1>numShape2)
                numShape1--;
            else
                numShape2--;
        }
        
    }

    public final void setDefaults(){
       seriesLength=DEFAULTSERIESLENGTH; 
       numShape1=DEFAULTNUMSHAPELETS;
       shapeletLength=DEFAULTSHAPELETLENGTH;
    }
    public ShapeType getShape(){
        return shape1.type;
    }
    public void setShapeType(ShapeType st){
        shape1.setType(st);
    }
   

//This constructor is used for data of a given length
    public void setNumShapes(int n){
        numShape1=n;
    }
    //This constructor is used for data of a given length in a two class problem
    //where the shape1 distinguishing the first class is known
  
    final public boolean setNonOverlappingLocations(){
        if(seriesLength-shapeletLength*totalNumShapes<totalNumShapes)  //Cannot fit them in, not enough spaces
          return false;
//Find non overlapping locations. Specify how many spaces there are
        int spaces=seriesLength-shapeletLength*totalNumShapes;
        int[] shapeLocations=new int[totalNumShapes];
        int[]  locs=new int[spaces];
        for(int i=0;i<spaces;i++)
            locs[i]=i;
        for(int i=0;i<spaces;i++){
            int a=Model.rand.nextInt(spaces);
            int b=Model.rand.nextInt(spaces);
            int temp=locs[a];
            locs[a]=locs[b];
            locs[b]=temp;
        }
            
       for(int i=0;i<totalNumShapes;i++){
           shapeLocations[i]=locs[i];
       }
/* Not worth the hassle
       //Sample without replacement the numShape1 start point using the Knuth algorithm
//looks like it is O(n) rather than O(nlogn) 
        int t = 0; // total input records dealt with
        int m = 0; // number of items selected so far
        double u;
        while (m < spaces){
            u = Model.rand.nextDouble(); // call a uniform(0,1) random number generator

            if ( (numShape1- t)*u >= spaces- m ){
                t++;
            }
            else {
                shapeLocations[m] = t;
                t++; m++;
            }
        }
        }
*/
//Split randomised locations for two types of shape
       for(int i=0;i<numShape1;i++)
           shape1Locations[i]=shapeLocations[i];
       for(int i=0;i<numShape2;i++)
           shape2Locations[i]=shapeLocations[i+numShape1];
       
       Arrays.sort(shape1Locations);
       Arrays.sort(shape2Locations);
//Find the positions by reflating
        int count1=0;
        int count2=0;
        for(int i=0;i<totalNumShapes;i++){
            if(count2==shape2Locations.length) //Finished shape2, must do shape1
               shape1Locations[count1++]+=i*shapeletLength;
            else if(count1==shape1Locations.length) //Finished shape1, must do shape1
               shape2Locations[count2++]+=i*shapeletLength;
            else if(shape1Locations[count1]<shape2Locations[count2])//Shape 1 before Shape 2, inflate shape 1
               shape1Locations[count1++]+=i*shapeletLength;
            else //Inflate shape2
               shape2Locations[count2++]+=i*shapeletLength;
        }
        return true;
    }
    
    /*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/
    @Override
	public double generate(double x){
//Noise
            int t=(int)x;

            double value=error.simulate();
//Shape: Check if in a shape1
 /*           int insertionPoint=Arrays.binarySearch(shapeLocations,t);
                        if(insertionPoint<0)//Not a start pos: in
                insertionPoint=-(1+insertionPoint);
//Too much grief, just doing a linear scan!            
            */
//See if it is in shape1            
            int insertionPoint=0;
            while(insertionPoint<shape1Locations.length && shape1Locations[insertionPoint]+shapeletLength<t)
                insertionPoint++;    
            if(insertionPoint>=shape1Locations.length){ //Bigger than all the start points, set to last
                insertionPoint=shape1Locations.length-1;
            }
            if(shape1Locations[insertionPoint]<=t && shape1Locations[insertionPoint]+shapeletLength>t){//in shape1
                value+=shape1.generateWithinShapelet(t-shape1Locations[insertionPoint]);
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);
            }else{  //Check if in shape 2
                insertionPoint=0;
                while(insertionPoint<shape2Locations.length && shape2Locations[insertionPoint]+shapeletLength<t)
                    insertionPoint++;    
                if(insertionPoint>=shape2Locations.length){ //Bigger than all the start points, set to last
                    insertionPoint=shape2Locations.length-1;
                }
                if(shape2Locations[insertionPoint]<=t && shape2Locations[insertionPoint]+shapeletLength>t){//in shape2
                    value+=shape2.generateWithinShapelet(t-shape2Locations[insertionPoint]);
//                System.out.println(" IN SHAPE 2 occurence "+insertionPoint+" Time "+t);
                }
            }
            return value;
        }

//This will generateWithinShapelet the next sequence after currently stored t value
    @Override
	public double generate()
        {
//            System.out.println("t ="+t);
            double value=generate(t);
            t++;
            return value;
        }
     @Override   
	public	double[] generateSeries(int n)
	{
           t=0;
//Resets the starting locations each time this is called          
           setNonOverlappingLocations();
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
    
   
    
    /**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */ 
    @Override
    public void setParameters(double[] param){
        if(param!=null){
            switch(param.length){
                default: 
                case 4:             shapeletLength=(int)param[3];
                case 3:             numShape2=(int)param[2];
                case 2:             numShape1=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
         
        
    }
    @Override
    public String getModelType(){ return "DictionarySimulator";}
    @Override
        public String getAttributeName(){return "Dict";} 
    @Override
        public String getHeader(){
            String header=super.getHeader();
            header+="%  \t Shapelet Length ="+shapeletLength;
            header+="\n%  \t Series Length ="+seriesLength;
            header+="\n%  \t Number of Shapelets ="+numShape1;
            header+="\n% \t Shape = "+shape1.type;
            return header;
        }
    
    
        
    // Inner class determining the shape1 inserted into the shapelet model
    public static class Shape{
        // Type: head and shoulders, spike, step, triangle, or sine wave.
        private ShapeType type;
        //Length of shape1
        private int length;
        //Position of shape1 on axis determined by base (lowest point) and amp(litude).
        private double base;
        private double amp;
        //The position in the series at which the shape1 begins.
        private int location;
        
        private static int DEFAULTBASE=-2;
        private static int DEFAULTAMP=4;
        
        //Default constructor, call randomise shape1 to get a random instance
        // The default length is 29, the shape1 extends from -2 to +2, is of 
        // type head and shoulders, and is located at index 0.
        private Shape()
        {
            this(ShapeType.HEADSHOULDERS,DEFAULTSHAPELETLENGTH,DEFAULTBASE,DEFAULTAMP); 
        }  
        //Set length only, default for the others
         private Shape(int length){
            this(ShapeType.HEADSHOULDERS,length,DEFAULTBASE,DEFAULTAMP);      
             
         }       
        // This constructor produces a completely specified shape1
        private Shape(ShapeType t,int l, double b, double a){
            type=t;
            length=l;
            base=b;
            amp=a;
        }
        
//Generates the t^th shapelet position
        private double generateWithinShapelet(int offset){
            double value=0;
            switch(type){
                 case TRIANGLE:
                    if(offset<=length/2) 
                          value=((offset/(double)(length/2))*(amp))+base;
                    else
                         value=((length-offset-1)/(double)(length/2)*(amp))+base;
                break;
                case HEADSHOULDERS:
                    if(offset<length/3)
                        value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*offset))+base;
                    else{
                        if(offset+1>=(2*length)/3)
                                value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset+1-(2*length)/3)))+base;
                        else
                            value = ((amp)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset-length/3)))+base;
                    }
                break;
                case SINE:
                     value=amp*Math.sin(((2*Math.PI)/(length-1))*offset)/2;
                break;
                case STEP:
                    if(offset<length/2)
                        value=base;
                    else
                        value=base+amp;
                    break;
                case SPIKE:
                    if(offset<=length/4){
                        if(offset==0)
                           value=0;
                        else
                           value=offset/(double)(length/4)*(-amp/2);
                    }
                    else if(offset>length/4 && offset<=length/2){
                        if(offset == length/2)
                            value=0;
                        else
                            value=(-amp/2)+((length/4-offset-1)/(double)(length/4)*(-amp/2));
                    }                               
                    else if(offset>length/2&&offset<=length/4*3)
                        value=(offset-length/2)/(double)(length/4)*(amp/2);
                    else if(offset>length/4*3)
                    {
                        if(offset+1==length)
                            value=0;
                        else
                            value=(length-offset-1)/(double)(length/4)*(amp/2);
                    }
                break;
            }
            return value;
        }
        
        
        public void setType(ShapeType newType){
            this.type=newType;
        }
        
        
        @Override
        public String toString()
        {
            String shp = ""+this.type+" start = "+location+" length ="+length;
            return shp;
        }
        
        //gives a shape1 a random type and start position
        private void randomiseShape(){
            ShapeType [] types = ShapeType.values();            
            int ranType = Model.rand.nextInt(types.length);
            setType(types[ranType]);
                
        }
    
}
    
    
    //Test harness
   
    public static void main (String[] args) throws IOException
    {
       Model.setDefaultSigma(0);
       Model.setGlobalRandomSeed(0);
       DEFAULTSHAPELETLENGTH=11;
       DEFAULTSERIESLENGTH=100;
       double[][] d=new double[ShapeType.values().length][DEFAULTSERIESLENGTH];
       int j=0;
       for(ShapeType s:ShapeType.values()){
           DEFAULTSHAPELETLENGTH+=2;
           DictionaryModel shape = new DictionaryModel();
           shape.setShapeType(s);
           for(int i=0;i<DEFAULTSERIESLENGTH;i++)
                d[j][i]=shape.generate(i);
           j++;
        }
    
       OutFile out=new OutFile("C:\\temp\\dictionaryModelTest.csv");
      for(int i=0;i<DEFAULTSERIESLENGTH;i++){
          for(j=0;j<d.length;j++)
            out.writeString(d[j][i]+",");
          out.writeString("\n");
      }
//       shape1.reset();
       
//       for(int i=0;i<200;i++)
//           System.out.println(shape1.generate(i));

         
    }
        
    
}
