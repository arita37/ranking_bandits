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
