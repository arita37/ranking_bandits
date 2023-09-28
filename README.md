<!-- Our title -->
<div align="center">
  <h3>ranking_bandits </h3>
</div>

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




1) run the example in the repo : only TopRANK

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







