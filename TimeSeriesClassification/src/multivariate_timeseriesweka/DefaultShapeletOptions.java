/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactoryOptions;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class DefaultShapeletOptions {
    
    public static final Map<String, Function<Instances, ShapeletTransformFactoryOptions>> FACTORY_OPTIONS;
    static {
        Map<String, Function<Instances, ShapeletTransformFactoryOptions>> map = new HashMap();
        map.put("INDEPENDENT", DefaultShapeletOptions::createIndependentShapeletSearch);
        map.put("SHAPELET_I", DefaultShapeletOptions::createSHAPELET_I);
        map.put("SHAPELET_D", DefaultShapeletOptions::createSHAPELET_D);

        FACTORY_OPTIONS = Collections.unmodifiableMap(map);
    }
    
    public static ShapeletTransformFactoryOptions createIndependentShapeletSearch(Instances train){
        ShapeletSearchOptions sOps = new ShapeletSearchOptions.Builder()
                                    .setMin(3)
                                    .setMax(utilities.MultivariateInstanceTools.channelLength(train))
                                    .setSearchType(ShapeletSearch.SearchType.MULTI_I)
                                    .build();

        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setSearchOptions(sOps)
                                            .setDistanceType(SubSeqDistance.DistanceType.DIMENSION)
                                            .setKShapelets(train.numInstances())
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
        
    public static ShapeletTransformFactoryOptions createSHAPELET_I(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(SubSeqDistance.DistanceType.INDEPENDENT)
                                            .setKShapelets(train.numInstances())
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSHAPELET_D(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(SubSeqDistance.DistanceType.DEPENDENT)
                                            .setKShapelets(train.numInstances())
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
}
