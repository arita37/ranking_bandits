
```python
#To run model 
python algo.py test2


All item [0 1 2 3 4 5 6 7 8 9]
Top k items [3 4 0 1 2]
Updating Batch Reward list and context
-------------Complete-----------
Reward model save


All item [0 1 2 3 4 5 6 7 8 9]
Top k items [0 1 2 4 3]
Updating Batch Reward list and context
-------------Complete-----------
Reward model save


All item [0 1 2 3 4]
Top k items [0 1 4 3 2]
Updating Batch Reward list and context
-------------Complete-----------
Reward model save


All item [0 1 2 3 4]
Input all reward list [array([[0.05375237]]), array([[0.00578542]]), array([[0.04082461]]), array([[0.02751526]]), array([[0.02033958]])]

Exploitation reward [array([[0.05375237]]), array([[0.04082461]])]

Expolaration : 3, so the expolaration rewrds [array([[0.02751526]]), array([[0.02033958]]), array([[0.00578542]])]

Top k items [0 1 2 4 3]
Updating Batch Reward list and context
-------------Complete-----------
Reward model save

All item [0 1 2 3 4]
Input all reward list [array([[0.06412207]]), array([[0.01080819]]), array([[-0.01225578]]), array([[0.05113878]]), array([[-0.07653962]])]

Exploitation reward [array([[0.06412207]]), array([[0.05113878]])]

Expolaration : 3, so the expolaration rewrds [array([[0.01080819]]), array([[-0.01225578]]), array([[-0.07653962]])]

Top k items [0 1 2 3 4]
Updating Batch Reward list and context
------------------------------------------------------------------------------------
All item [0 1 2 3 4]
Input all reward list [array([[0.02954466]]), array([[-0.01319769]]), array([[0.01170897]]), array([[0.10332213]]), array([[0.01760611]])]
After sorting item id on the basis of best reward [3 0 4 2 1]

Exploitation reward: [array([[0.10332213]]), array([[0.02954466]])] and item id [3 0]

Expolaration : 3, so the expolaration rewrds [array([[0.01760611]]), array([[0.01170897]]), array([[-0.01319769]])]
Items need to explore [4 2 1]

Top k items [3 0 1 4 2]
Updating Batch Reward list and context
Before Batch Update 

After Batch Update
-------------Complete-----------
Reward model save

-------------------------------------------------------------------------------------
All item [0 1 2 3 4]
Input all reward list [array([[-0.00420114]]), array([[0.00695873]]), array([[0.01841998]]), array([[-0.02038802]]), array([[-0.0608413]])]
After sorting item id on the basis of best reward [2 1 0 3 4]

Exploitation reward: [array([[0.01841998]]), array([[0.00695873]])] and item id [2 1]

Expolaration : 3, so the expolaration rewrds [array([[-0.00420114]]), array([[-0.02038802]]), array([[-0.0608413]])]
Items need to explore [0 3 4]

Top k items [2 1 4 0 3]
Updating Batch Reward list and context
Before Batch Update 

After Batch Update
-------------Complete-----------
Reward model save










--------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
All item [0 1 2 3 4 5 6]
Input all reward list [array([[0.79708402]]), array([[-0.16076523]]), array([[-0.94905603]]), array([[1.51728249]]), array([[0.11650817]]), array([[1.71635468]]), array([[-0.30406995]])]
After sorting item id on the basis of best reward [5 3 0 4 1 6 2]

Exploitation reward: [array([[1.71635468]]), array([[1.51728249]])] and item id [5 3]

Expolaration : 5, so the expolaration rewrds [array([[0.79708402]]), array([[0.11650817]]), array([[-0.16076523]]), array([[-0.30406995]]), array([[-0.94905603]])]
Items need to explore [0 4 1 6 2] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-0.48119101]]), array([[-1.05126058]]), array([[-0.17903715]]), array([[-0.6414572]]), array([[-0.71637917]]), array([[1.23332544]]), array([[1.05315404]])]
After sorting item id on the basis of best reward [5 6 2 0 3 4 1]

Exploitation reward: [array([[1.23332544]]), array([[1.05315404]])] and item id [5 6]

Expolaration : 5, so the expolaration rewrds [array([[-0.17903715]]), array([[-0.48119101]]), array([[-0.6414572]]), array([[-0.71637917]]), array([[-1.05126058]])]
Items need to explore [2 0 3 4 1] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-0.04253533]]), array([[1.72785503]]), array([[-1.7099122]]), array([[0.53315696]]), array([[1.14068455]]), array([[2.14910277]]), array([[-0.2500961]])]
After sorting item id on the basis of best reward [5 1 4 3 0 6 2]

Exploitation reward: [array([[2.14910277]]), array([[1.72785503]])] and item id [5 1]

Expolaration : 5, so the expolaration rewrds [array([[1.14068455]]), array([[0.53315696]]), array([[-0.04253533]]), array([[-0.2500961]]), array([[-1.7099122]])]
Items need to explore [4 3 0 6 2] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-2.15550776]]), array([[-1.27829548]]), array([[-0.21792101]]), array([[-0.81580483]]), array([[0.62799745]]), array([[-1.90740382]]), array([[-0.00391614]])]
After sorting item id on the basis of best reward [4 6 2 3 1 5 0]

Exploitation reward: [array([[0.62799745]]), array([[-0.00391614]])] and item id [4 6]

Expolaration : 5, so the expolaration rewrds [array([[-0.21792101]]), array([[-0.81580483]]), array([[-1.27829548]]), array([[-1.90740382]]), array([[-2.15550776]])]
Items need to explore [2 3 1 5 0] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-1.39558205]]), array([[-1.77728564]]), array([[-2.49435673]]), array([[-1.02330269]]), array([[1.22089601]]), array([[-0.93573751]]), array([[2.03341214]])]
After sorting item id on the basis of best reward [6 4 5 3 0 1 2]

Exploitation reward: [array([[2.03341214]]), array([[1.22089601]])] and item id [6 4]

Expolaration : 5, so the expolaration rewrds [array([[-0.93573751]]), array([[-1.02330269]]), array([[-1.39558205]]), array([[-1.77728564]]), array([[-2.49435673]])]
Items need to explore [5 3 0 1 2] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-1.46044026]]), array([[-0.50365433]]), array([[1.13037833]]), array([[0.29074555]]), array([[-1.63245275]]), array([[-0.88570539]]), array([[-0.58089019]])]
After sorting item id on the basis of best reward [2 3 1 6 5 0 4]

Exploitation reward: [array([[1.13037833]]), array([[0.29074555]])] and item id [2 3]

Expolaration : 5, so the expolaration rewrds [array([[-0.50365433]]), array([[-0.58089019]]), array([[-0.88570539]]), array([[-1.46044026]]), array([[-1.63245275]])]
Items need to explore [1 6 5 0 4] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-0.90582286]]), array([[2.49688122]]), array([[0.22118202]]), array([[0.51229532]]), array([[-0.52104323]]), array([[-0.19074793]]), array([[0.4305956]])]
After sorting item id on the basis of best reward [1 3 6 2 5 4 0]

Exploitation reward: [array([[2.49688122]]), array([[0.51229532]])] and item id [1 3]

Expolaration : 5, so the expolaration rewrds [array([[0.4305956]]), array([[0.22118202]]), array([[-0.19074793]]), array([[-0.52104323]]), array([[-0.90582286]])]
Items need to explore [6 2 5 4 0] 

All item [0 1 2 3 4 5 6]
Input all reward list [array([[-0.07374619]]), array([[-0.56193254]]), array([[0.71858272]]), array([[0.73434248]]), array([[-0.34500468]]), array([[-0.14536724]]), array([[0.41890078]])]
After sorting item id on the basis of best reward [3 2 6 0 5 4 1]

Exploitation reward: [array([[0.73434248]]), array([[0.71858272]])] and item id [3 2]

Expolaration : 5, so the expolaration rewrds [array([[0.41890078]]), array([[-0.07374619]]), array([[-0.14536724]]), array([[-0.34500468]]), array([[-0.56193254]])]
Items need to explore [6 0 5 4 1] 



metric creation dataframe     ts            itemid_list             itemid_clk  ...  regret regret_bad_cum  regret_ratio
0    0  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 1, 0, 1, 1]  ...     0.0            0.0           0.0
1    1  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 1, 1]  ...     0.0            0.0           0.0
2    2  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 0, 1]  ...     0.0            0.0           0.0
3    3  [0, 1, 2, 3, 4, 5, 6]  [0, 1, 0, 0, 0, 1, 1]  ...     0.0            0.0           0.0
4    4  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 1, 1]  ...     0.0            0.0           0.0
5    5  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 0, 1, 1]  ...     0.0            0.0           0.0
6    6  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 0, 1, 0]  ...     0.0            0.0           0.0
7    7  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 1, 1]  ...     0.0            0.0           0.0
8    8  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 1, 1]  ...     0.0            0.0           0.0
9    9  [0, 1, 2, 3, 4, 5, 6]  [0, 0, 0, 0, 1, 1, 1]  ...     0.0            0.0           0.0
10   0  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
11   1  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
12   2  [0, 1, 2, 3, 4, 5, 6]  [1, 0, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
13   3  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
14   4  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
15   5  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 0, 0, 0, 0, 0]  ...     0.0            0.0           0.0
16   6  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
17   7  [0, 1, 2, 3, 4, 5, 6]  [1, 0, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
18   8  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0
19   9  [0, 1, 2, 3, 4, 5, 6]  [1, 1, 1, 0, 0, 0, 0]  ...     0.0            0.0           0.0

[20 rows x 12 columns]
    rwd_best  rwd_actual  regret_cum  regret_bad_cum  regret_ratio
0          3         3.0         0.0             0.0           0.0
1          3         3.0         0.0             0.0           0.0
2          2         2.0         0.0             0.0           0.0
3          3         3.0         0.0             0.0           0.0
4          3         3.0         0.0             0.0           0.0
5          2         2.0         0.0             0.0           0.0
6          1         1.0         0.0             0.0           0.0
7          3         3.0         0.0             0.0           0.0
8          3         3.0         0.0             0.0           0.0
9          3         3.0         0.0             0.0           0.0
10         3         3.0         0.0             0.0           0.0
11         3         3.0         0.0             0.0           0.0
12         2         2.0         0.0             0.0           0.0
13         3         3.0         0.0             0.0           0.0
14         3         3.0         0.0             0.0           0.0
15         2         2.0         0.0             0.0           0.0
16         3         3.0         0.0             0.0           0.0
17         2         2.0         0.0             0.0           0.0
18         3         3.0         0.0             0.0           0.0
19         3         3.0         0.0             0.0           0.0


ztmp/exp//20231128_2220_T10/sim0/metrics/simul_metrics.csv
    ts    itemid_list     itemid_clk  context  ... rwd_actual_cum  regret_cum  regret_bad_cum_cum regret_ratio_cum
0    0  0,1,2,3,4,5,6  0,0,0,1,0,1,1        0  ...            3.0         0.0                 0.0              0.0
1    1  0,1,2,3,4,5,6  0,0,0,0,1,1,1        0  ...            6.0         0.0                 0.0              0.0
2    2  0,1,2,3,4,5,6  0,0,0,0,1,0,1        0  ...            8.0         0.0                 0.0              0.0
3    3  0,1,2,3,4,5,6  0,1,0,0,0,1,1        0  ...           11.0         0.0                 0.0              0.0
4    4  0,1,2,3,4,5,6  0,0,0,0,1,1,1        0  ...           14.0         0.0                 0.0              0.0
5    5  0,1,2,3,4,5,6  0,0,0,0,0,1,1        0  ...           16.0         0.0                 0.0              0.0
6    6  0,1,2,3,4,5,6  0,0,0,0,0,1,0        0  ...           17.0         0.0                 0.0              0.0
7    7  0,1,2,3,4,5,6  0,0,0,0,1,1,1        0  ...           20.0         0.0                 0.0              0.0
8    8  0,1,2,3,4,5,6  0,0,0,0,1,1,1        0  ...           23.0         0.0                 0.0              0.0
9    9  0,1,2,3,4,5,6  0,0,0,0,1,1,1        0  ...           26.0         0.0                 0.0              0.0
10   0  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           29.0         0.0                 0.0              0.0
11   1  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           32.0         0.0                 0.0              0.0
12   2  0,1,2,3,4,5,6  1,0,1,0,0,0,0        1  ...           34.0         0.0                 0.0              0.0
13   3  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           37.0         0.0                 0.0              0.0
14   4  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           40.0         0.0                 0.0              0.0
15   5  0,1,2,3,4,5,6  1,1,0,0,0,0,0        1  ...           42.0         0.0                 0.0              0.0
16   6  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           45.0         0.0                 0.0              0.0
17   7  0,1,2,3,4,5,6  1,0,1,0,0,0,0        1  ...           47.0         0.0                 0.0              0.0
18   8  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           50.0         0.0                 0.0              0.0
19   9  0,1,2,3,4,5,6  1,1,1,0,0,0,0        1  ...           53.0         0.0                 0.0              0.0


```




#Update 

After sorting item id on the basis of best reward [6 1 5 0 4 2 3]
Actions [0 1 2 3 4 5 6] and its location is 0
After sorting item id on the basis of best reward [5 0 6 2 4 1 3]
Actions [5 6 4 3 2 1 0] and its location is 1

