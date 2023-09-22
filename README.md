<!-- Our title -->
<div align="center">
  <h3>ranking_bandits </h3>
</div>

<!-- Short description -->
<p align="center">
A Python-toolkit for bandit algorithms choosing a ranked-list of $K$ items among $L$ at each iteration
</p>



```
TODO

Bandit Recommendation at algo: 

at each time step

    t0:    bandit select top -k items out of L items.

    t2:    bandit select top -k items out of L items.     
               --->  Next step, check which item is clicked. --> reward --> update the algo.

    t3:    bandit select top -k items out of L items.


   5 or 6 algos: in this repo,
   interest only in one:   TopRank





1) run the example in the repo :

    https://github.dev/arita37/ranking_bandits/blob/main/README.md

    test_exp.sh

       python3 exp.py --play 2 100 -r 10 --small --TopRank 100  --horizon_time_known $TESTDIR/dev_null --force || exit
       python3 exp.py --merge 100 -r 10 --small  --TopRank 100  --horizon_time_known $TESTDIR/dev_null || exit




2) Plug the dataset : (CSV file)

   time, locationid,  itemid, is_click
                                 1
   each locationid as L itemid
       Recommend top-K items for each locationid


  Goal is to load/transform into the correct format and run the algo TopRank.
     map the CSV into correct JSON format










```




<!-- Draw horizontal rule -->
<hr>

> This contains the code used for the research articles
> * [Parametric Graph for Unimodal Ranking Bandit]() (Camille-Sovaneary Gauthier, Romaric Gaudel, Elisa Fromont, and Boammani Aser Lompo,  2021) presented at the 38$^{th}$ International Conference on Machine Learning ([ICML'21](https://icml.cc/Conferences/2021)).
> * [Bandit Algorithm for Both Unknown BestPosition and Best Item Display on Web Pages]() (Camille-Sovaneary Gauthier, Romaric Gaudel, and Elisa Fromont, 2021) presented at the 19$^{th}$ Symposium on Intelligent Data Analysis ([IDA'21](https://ida2021.org/)).


## Requirements

* Python >= 3.6
* libraries (see `requirements.txt` build with Python 3.8.7)
    * numpy
    * scipy
    * (for PMED) tensorflow 
    * (for `exp.py` and `run_experiments.py`) docopt
    * (for `run_experiments.py`) matplotlib

## (Re)run experiments

See corresponding README file:

* [README_IDA_2021.md](README_IDA_2021.md)
* [README_ICML_2021.md](README_ICML_2021.md)


```bibtex
  pdf = 	 {http://proceedings.mlr.press/v139/gauthier21a/gauthier21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/gauthier21a.html}
}
```



