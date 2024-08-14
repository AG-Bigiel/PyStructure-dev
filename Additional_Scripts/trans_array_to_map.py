import numpy as np

def array_to_map(ra,dec,val,nearest=False,cutoff=3.0,kernel=10.0,gridspacing=1.0):

    res=gridspacing/3600.

    # define new high-res coordinate axis (newx,newy)
    xborder=(2.5*cutoff*kernel/3600.)*np.cos(np.deg2rad(np.mean(dec)))
    yborder=2.5*cutoff*kernel/3600.
    nra  = int(round(  (( ra.max()-ra.min()+xborder ) * np.cos(np.deg2rad(np.mean(dec))))  / res ,0))
    ndec = int(round(  (dec.max()-dec.min()+yborder )   / res ,0))
    newx=np.linspace(ra.min()-xborder/2,ra.max()+xborder/2,endpoint=True,num=nra)
    newy=np.linspace(dec.min()-yborder/2,dec.max()+yborder/2,endpoint=True,num=ndec)
    # create meshgrid
    newx,newy=np.meshgrid(newx,newy,indexing='xy')

    # size of new gird
    w=newx.shape[1]
    h=newy.shape[0]

    ###############################################
    # Nearest neighbour interpolation
    ###############################################

    if nearest:
        datamap = np.array([[None for a in range(w)] for b in range(h)],dtype=float)

        for i in range(len(ra)):
            rai=ra[i]
            dei=dec[i]

            distances = np.sqrt( ((newx-rai)*np.cos(np.deg2rad(dei)))**2.0 + (newy-dei)**2.0 )

            ind=np.where(distances < cutoff*kernel/3600.)

            datamap[ind] = val[i]

            del distances, ind

        return datamap,newx,newy

    ###############################################
    # Convolution with gridding kernel
    ###############################################

    else:
        datamap = np.array([[0.0 for a in range(w)] for b in range(h)],dtype=float)
        weightmap = np.array([[0.0 for a in range(w)] for b in range(h)],dtype=float)

        for i in range(len(ra)):
            rai=ra[i]
            dei=dec[i]

            distances = np.sqrt( ((newx-rai)*np.cos(np.deg2rad(dei)))**2.0 + (newy-dei)**2.0 )

            ind=np.where(distances < cutoff*kernel/3600.)

            weights = np.exp(-1.*distances[ind]**2/2./((kernel/3600.)/2.355)**2)
            datamap[ind] += val[i] * weights
            weightmap[ind] += weights

            del distances, weights, ind

        weightmap[weightmap==0.0]=None
        datamap=datamap/weightmap

        return datamap,newx,newy
