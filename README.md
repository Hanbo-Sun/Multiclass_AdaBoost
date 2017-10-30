# Multi-class-AdaBoost
SAMME - Stagewise Additive Modeling  using a Multi-class Exponential loss function 

This project implemented a novel Multi-class AdaBoost Algorithm referred as SAMME [1] – Stagewise Additive Modeling using a Multi-class Exponential loss function. By adding the log(K + 1) (K denotes classes) term to penalize weak classifier, SAMME not only overcomes the restriction barrier of handling multi-class problems in AdaBoost.M1, but also could be generalized into any fisher-consistent loss function. SAMME is equivalent to stagewise additive modeling. Also, the choice of multi-class exponential loss function satisfying Bayes rule. 

A tests were performaned on UC-Irvine machine learning database to compare the performance of SAMME.DT(SAMME based on weak classifier Decision Tree), and SAMME.NB(SAMME based on Naive Bayes), decision tree, Naive Bayes, and AdaBoost.M1. SAMME.DT outperforms others.


References
[1] Ji Zhu. & Hui Zou. & Saharon Rosset. & Trevor Hastie. (2009) Multi-class Adaboost. Statistics and Its Inter- face, Volume 2 (2009), pp. 349-360.
