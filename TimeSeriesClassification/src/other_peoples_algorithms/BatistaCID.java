/*
Implementation of the algorithm described in 

@inproceedings{batista11cid,
author="G. Batista and X. Wang and E. Keogh ",
title="A Complexity-Invariant Distance Measure for Time Series",
booktitle    ="Proceedings of the 11th {SIAM} International Conference on Data Mining (SDM)",
year="2011"
}
and 
@inproceedings{batista14cid,
author="G. Batista and E. Keogh and O. Tataw and X. Wang  ",
title="{CID}: an efficient complexity-invariant distance for time series",
  journal={Data Mining and Knowledge Discovery},
  volume={28},
  pages="634--669",
  year={2014}
}

The distance measure CID(Q,C)=ED(Q,C) × CF(Q,C), 
where ED is the Eucidean distance and
CF(Q,C) = max (CE(Q),CE(C))
          min (CE(Q),CE(C)) 
ie the ratio of complexities. In thepaper, 

*/
package other_peoples_algorithms;

/**
 *
 * @author ajb
 */
public class BatistaCID {
    
}
