# Extracted rule base: mackey_glass

Model: 16 rules, 96 parameters (16 premise + 80 consequent), gaussian membership functions.

Rules needed to account for 90% of the output: 12 of 16.
Effective rules (firing perplexity): 12.4. Dead rules: 0.

## Top rules by output share

```
R 12  IF x(t-18) is high (~3.01) AND x(t-12) is high (~2.59) AND x(t-6) is low (~-2.07) AND x(t) is low (~-2.01)
      THEN x(t+6) = -0.367*x(t-18) + -0.505*x(t-12) + +0.831*x(t-6) + -0.350*x(t) -1.614
      (fires on 14.6% of mass, 19.2% of output)
R  7  IF x(t-18) is low (~-2.95) AND x(t-12) is high (~2.59) AND x(t-6) is high (~2.84) AND x(t) is high (~2.82)
      THEN x(t+6) = -1.433*x(t-18) + -0.513*x(t-12) + -4.812*x(t-6) + +3.466*x(t) +3.517
      (fires on 6.4% of mass, 13.9% of output)
R  0  IF x(t-18) is low (~-2.95) AND x(t-12) is low (~-2.75) AND x(t-6) is low (~-2.07) AND x(t) is low (~-2.01)
      THEN x(t+6) = +0.252*x(t-18) + +0.696*x(t-12) + +0.064*x(t-6) + +0.592*x(t) +2.113
      (fires on 17.8% of mass, 11.3% of output)
R  5  IF x(t-18) is low (~-2.95) AND x(t-12) is high (~2.59) AND x(t-6) is low (~-2.07) AND x(t) is high (~2.82)
      THEN x(t+6) = +3.186*x(t-18) + -3.860*x(t-12) + +1.654*x(t-6) + -6.799*x(t) +6.186
      (fires on 5.0% of mass, 10.2% of output)
R 13  IF x(t-18) is high (~3.01) AND x(t-12) is high (~2.59) AND x(t-6) is low (~-2.07) AND x(t) is high (~2.82)
      THEN x(t+6) = +5.561*x(t-18) + -5.584*x(t-12) + +0.520*x(t-6) + -0.170*x(t) -3.797
      (fires on 2.3% of mass, 7.5% of output)
R 14  IF x(t-18) is high (~3.01) AND x(t-12) is high (~2.59) AND x(t-6) is high (~2.84) AND x(t) is low (~-2.01)
      THEN x(t+6) = +0.321*x(t-18) + -0.391*x(t-12) + -0.556*x(t-6) + +1.371*x(t) +0.088
      (fires on 8.8% of mass, 4.9% of output)
R  1  IF x(t-18) is low (~-2.95) AND x(t-12) is low (~-2.75) AND x(t-6) is low (~-2.07) AND x(t) is high (~2.82)
      THEN x(t+6) = +0.227*x(t-18) + +0.636*x(t-12) + -0.425*x(t-6) + +1.037*x(t) +0.821
      (fires on 9.3% of mass, 4.8% of output)
R  3  IF x(t-18) is low (~-2.95) AND x(t-12) is low (~-2.75) AND x(t-6) is high (~2.84) AND x(t) is high (~2.82)
      THEN x(t+6) = +0.148*x(t-18) + +0.872*x(t-12) + -1.533*x(t-6) + +1.828*x(t) +0.866
      (fires on 5.4% of mass, 4.8% of output)
```

## Local explanation for one held-out prediction

Prediction (standardised): -0.1043

|   rule | IF                                                                    |   firing_strength |   contribution |   pct_of_prediction |
|-------:|:----------------------------------------------------------------------|------------------:|---------------:|--------------------:|
|      8 | x(t-18) is high AND x(t-12) is low AND x(t-6) is low AND x(t) is low  |         0.488847  |      -0.172663 |            -165.603 |
|      0 | x(t-18) is low AND x(t-12) is low AND x(t-6) is low AND x(t) is low   |         0.404482  |       0.114626 |             109.939 |
|     12 | x(t-18) is high AND x(t-12) is high AND x(t-6) is low AND x(t) is low |         0.0484193 |      -0.112617 |            -108.012 |
