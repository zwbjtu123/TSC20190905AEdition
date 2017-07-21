package timeseriesweka.filters.shapelet_transforms;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.class_value.BinaryClassValue;
import timeseriesweka.filters.shapelet_transforms.class_value.NormalClassValue;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchFactory;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import timeseriesweka.filters.shapelet_transforms.distance_functions.CachedSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.DimensionDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.ImprovedOnlineSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.MultivariateDependentDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.MultivariateIndependentDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.OnlineCachedSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.OnlineSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType;
import static timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.NORMAL;

/**
 *
 * @author Aaron
 */
public class ShapeletTransformFactory {
    
    
    //could change this to a map tbh.
    private static final List<Supplier<SubSeqDistance>> distanceFunctions = createDistanceTable();
    
    private static List<Supplier<SubSeqDistance>> createDistanceTable(){
        //istanceType{NORMAL, ONLINE, IMP_ONLINE, CACHED, ONLINE_CACHED, DEPENDENT, INDEPENDENT};
        List<Supplier<SubSeqDistance>> dCons = new ArrayList<>();
        dCons.add(SubSeqDistance::new);
        dCons.add(OnlineSubSeqDistance::new);
        dCons.add(ImprovedOnlineSubSeqDistance::new);
        dCons.add(CachedSubSeqDistance::new);
        dCons.add(OnlineCachedSubSeqDistance::new);
        dCons.add(MultivariateDependentDistance::new);
        dCons.add(MultivariateIndependentDistance::new);
        dCons.add(DimensionDistance::new);
        return dCons;
    }
    
    ShapeletTransformFactoryOptions options;
    public ShapeletTransformFactory(ShapeletTransformFactoryOptions op){
        options = op;
    }
    

    public ShapeletTransform getTransform(){
        //build shapelet transform based on options.
        ShapeletTransform st = createTransform(options.isBalanceClasses());
        st.setClassValue(createClassValue(options.isBinaryClassValue()));
        st.setShapeletMinAndMax(options.getMinLength(), options.getMaxLength());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setSubSeqDistance(createDistance(options.getDistance()));
        st.setSearchFunction(createSearch(options.getSearchOptions()));
        st.setQualityMeasure(options.getQualityChoice());
        st.setRoundRobin(options.useRoundRobin());
        st.setCandidatePruning(options.useCandidatePruning());
        return st;
    }    
    
    //time, number of cases, and length of series
    public static ShapeletTransform createDefaultTimedTransform(long numShapeletsToEvaluate, int n, int m, long seed){
        ShapeletSearchOptions sOp = new ShapeletSearchOptions.Builder()
                                        .setMin(3)
                                        .setMax(m)
                                        .setSearchType(ShapeletSearch.SearchType.IMP_RANDOM)
                                        .setNumShapelets(numShapeletsToEvaluate)
                                        .setSeed(seed)
                                        .build();
        
        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .useClassBalancing()
                                            .useBinaryClassValue()
                                            .useCandidatePruning()
                                            .setKShapelets(n)
                                            .setDistanceType(DistanceType.IMP_ONLINE)
                                            .setSearchOptions(sOp)
                                            .build();
        
        return new ShapeletTransformFactory(options).getTransform();
    }
    
    private ShapeletSearch createSearch(ShapeletSearchOptions sOp){
        return new ShapeletSearchFactory(sOp).getShapeletSearch();
    }
    
    private NormalClassValue createClassValue(boolean classValue){
        return classValue ?  new BinaryClassValue() : new NormalClassValue();
    }
    
    private ShapeletTransform createTransform(boolean balance){
        return balance ?  new BalancedClassShapeletTransform() : new ShapeletTransform();
    }
    
    private SubSeqDistance createDistance(DistanceType dist){
            return distanceFunctions.get(dist.ordinal()).get();
    }
    
    
    public static void main(String[] args) {
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                                    .useClassBalancing()
                                                    .setKShapelets(1000)
                                                    .setDistanceType(NORMAL)
                                                    .setMinLength(3)
                                                    .setMaxLength(100)
                                                    .build();
        
        ShapeletTransformFactory factory = new ShapeletTransformFactory(options);
        ShapeletTransform st = factory.getTransform();
    }
}
