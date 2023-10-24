""" 



"""
import networkx as nx
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import tqdm
import pathlib

from fastrec import GraphRecommender

from utilmy import (log, os_makedirs, pd_read_file, pd_to_file)


def animate(labelsnp, all_embeddings,mask, dirout="ztmp/"):
    labelsnp = labelsnp[mask]

    for i,embedding in enumerate(tqdm.tqdm(all_embeddings)):
        data = embedding[mask]
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        plt.title('Epoch {}'.format(i))

        colormap = ['r' if l=='Administrator' else 'b' for l in labelsnp]
        plt.scatter(data[:,0],data[:,1], c=colormap)

        ax.annotate('Administrator',(data[0,0],data[0,1]))
        ax.annotate('Instructor',(data[33,0],data[33,1]))

        plt.savefig('./ims/{n}.png'.format(n=i))
        plt.close()

    imagep = pathlib.Path( dirout + '/ims/')
    images = imagep.glob('*.png')
    images = list(images)
    images.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
    with imageio.get_writer('./animation.gif', mode='I') as writer:
        for image in images:
            data = imageio.imread(image.__str__())
            writer.append_data(data)


def run(epoch=2, debug=1, dirout="ztmp/"):
    """
    
     python karateclub.py run --epoch 1  --debug 0
    

    """
    g = nx.karate_club_graph()
    nodes = list(g.nodes)
    e1,e2 = zip(*g.edges)
    attributes = pd.read_csv('./karate_attributes.csv')

    sage = GraphRecommender(2,distance='l2')
    sage.add_nodes(nodes)
    sage.add_edges(e1,e2)
    sage.add_edges(e2,e1)
    sage.update_labels(attributes.community)

    ### Debug
    epochs, batch_size = epoch, 15
    if debug>0:
        loss_history,perf_history,all_embeddings  = sage.train(epochs, batch_size, unsupervised = True, 
                                learning_rate=1e-2, 
                                test_every_n_epochs=10, 
                                return_intermediate_embeddings=True)

        log(all_embeddings)
    else: 
        loss_history,perf_history = sage.train(epochs, batch_size, unsupervised = True, 
                                learning_rate=1e-2, 
                                test_every_n_epochs=10, 
                                return_intermediate_embeddings= False)

        all_embeddings = sage.embeddings

        all_embeddings = np_to_strlist(all_embeddings)
        dfnode = sage.node_ids
        dfnode['emb'] = all_embeddings
        pd_to_file(dfnode, dirout + "/dfemb.parquet", show=1)


    #animate(sage.labels,all_embeddings,sage.entity_mask)
    topk = sage.query_neighbors([0,33],k=5)
    


    # sage.start_api()


def np_to_strlist(v):
    v2 = []
    for vi in v: 
         si = ",".join([ str(xi) for xi in vi ] )
         v2.append(si)
    return v2


if __name__=='__main__':
   import fire  
   fire.Fire()