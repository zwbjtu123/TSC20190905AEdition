/*
TO DO PLAN:
    Sort out cluster BSUB settings.
    Run k-NN DTWCV on train test
    Run ED/DTW/DTWCV on resample
    

Evaluation plan
TimeDomainBenchmarks: 
    Write up: TimeDomainBenchmarks.tex
    Results:  TimeDomainBenchmarks.xlsx
Set 1: 
1. NN ED/DTW/DTWCV   Train/Test: All done
        1-NN vs k-NN
        Normalise vs not normalise
        setting window size
        
2. NN ED/DTW/DTWCV   Resample 100: Partial
        1-NN vs k-NN
        Normalise vs not normalise
        setting window size
3. Other classifiers on the raw data
Train/Test+ Resample
        Individual vs Weighted Ensemble
4. Feature selection: interval based. 
    Look at relationship between ED and DTW, whats the correlation? If good, use
    ED to choose interval. 
    Update arvx file, maybe send somewhere low level 
    Compare to stepwise/gavin method

SummarStatsBenchmarks
Run all classifiers on     

*/
package development;

/**
 *
 * @author ajb
 */
public class TSC_Evaluation {
    
}
