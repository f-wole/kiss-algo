This repository contains code to perform mean-reversion technique on an index.

Keep  
It  
Simple,  
Stupid  

Mean-reversion is performed using weighted exponential moving averages.
Code can be run using *tune_eval.sh*.

Future developments:  
1) add maximum loss to exit the market (problem: one must then enter the market in the next in signal (transition from negative to positive))
2) differentiate periods of moving averages for in and out signals