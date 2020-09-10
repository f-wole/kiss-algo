## Keep It Simple, Stupid

This repository contains code to perform mean-reversion technique on an index.
 
Mean-reversion is performed using weighted exponential moving averages. It is possible to set the maximum loss (over a period of 2 weeks) which triggers the exit from the market.

Code can be run using *tune_eval.sh*.

Future developments:  
- add another moving_average for further control: buy-signal: 10-period MA crosses over 50-period MA AND the price is above the 200-period MA. Similar for the sell-signal.
- differentiate periods of moving averages for in and out signals