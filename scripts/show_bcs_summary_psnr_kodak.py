#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('cairo')
import matplotlib.pyplot as plt

np.set_printoptions(precision=4,suppress=True)
# columns are
# 0 Rate (1 to 5 representing 10% to 50%)
# 1 patch width
# 2 (inverse) overlap factor: the overlap between patches is width/overlap
# 3 number of non overlapping samples in patch (effective degrees of freedom)
# 4 stride
# 5 compressed samples per patch
# 7 Kodak image number
# 8 random seed
# 9 RMSE
# 10 PSNR


# from pyplot doc:
# markers = {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'
#
import matplotlib.cm as cm

def plot_results(rate,widths,overlaps,z25,z50,z75,best):
    comap = cm.get_cmap('jet_r')
    plt.figure(figsize=(5,4))
    iov = (1/overlaps)
    s25 = 1000*(z25-22)/12
    s50 = 1000*(z50-22)/12
    s75 = 1000*(z75-22)/12
    imax   = np.argmax(z50)
    # mark best
    plt.scatter(widths[imax],iov[imax],marker="o",s=s75[imax]+300,facecolors="none",vmin=22,vmax=34,alpha=1,edgecolors="black",cmap=comap)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(ticks=[8,16,32],labels=['8x8','16x16','32x32'])
    plt.yticks(ticks=[1/32,1/16,1/8,1/4,1/2],labels=['1/32','1/16','1/8','1/4','1/2'])
    plt.xlim(8*np.power(2,-1/2),32*np.power(2,1/2))
    plt.ylim((1/32)*np.power(2,-1/2),(1/2)*np.power(2,1/2))
    plt.ylabel('overlap')
    plt.xlabel('block size')
    # show all
    plt.scatter(widths,iov,marker="o",s=s75,c=z50,vmin=22,vmax=34,alpha=0.33,edgecolors=None,cmap=comap)
    plt.scatter(widths,iov,marker="o",s=s50,c=z50,vmin=22,vmax=34,alpha=0.5,edgecolors="black",cmap=comap)
    plt.scatter(widths,iov,marker="o",s=s25,c=z50,vmin=22,vmax=34,alpha=1,edgecolors=None,cmap=comap)
    plt.colorbar()
    plt.title(f"RMSE for subsampling rate {10*rate}%")
    plt.savefig(f'rate{rate}.svg')
    plt.close('all')

#
#
# for each rate we generate a 2D matrix with the average RMSE of the recovered image
# for each value of width and stride
#
if __name__ == '__main__':
    plt.close('all')
    S = np.loadtxt('results/summary.csv')
    #
    # replace 7 lines with 9 different seeds by three columns: percentiles 25 50 75
    #
    n, m = S.shape
    images = np.unique(S[:,-4])
    print('images',images)
    seeds  = np.unique(S[:,-3])
    print('seeds',seeds)
    nseeds = len(seeds)
    nimages = len(images)
    n2 = n // nseeds
    S2 = np.empty((n2, m))
    for i in range(n2):
        S2[i, :-3] = S[i * nseeds, :-3]
        samples = S[i * nseeds:(i + 1) * nseeds, -2]  # RMSE for all seeds
        S2[i, -3] = np.percentile(samples, 25)
        S2[i, -2] = np.percentile(samples, 50)
        S2[i, -1] = np.percentile(samples, 75)
    #
    # do the same with Kodak images
    n3 = n // (nseeds*nimages)
    S3 = np.empty((n3, m - 1))
    for i in range(n3):
        S3[i, :-3] = S[i * nseeds*nimages, :-4]
        samples = S[i * nseeds*nimages:(i + 1) * nseeds*nimages, -2]  # RMSE for  7 seeds and 7 images
        S3[i, -3] = np.percentile(samples, 25)
        S3[i, -2] = np.percentile(samples, 50)
        S3[i, -1] = np.percentile(samples, 75)

    #
    # convert stride info into overlap factor: stride = w - w/of -> of = w/(w - stride)
    #
    S3[:,2] = np.round(S3[:,1]/(S3[:,1]-S3[:,2]))
    med  = np.zeros(5)
    yerr = np.zeros((5,2))
    rates = np.arange(1,5+1)
    for r,rate in enumerate(rates):
        k = np.flatnonzero(S3[:, 0] == rate)
        S4 = S3[k, :]
        w = S4[:,1]
        o = S4[:,2]
        e25 = S4[:, -3] # this is better than e75!! smaller RMSE is better
        e50 = S4[:, -2]
        e75 = S4[:, -1]
        ibest = np.argmin(e50)
        z25 = -20 * np.log10(e75) # PSNR inverts -> percentiles switch!
        z50 = -20 * np.log10(e50) # stays the same
        z75 = -20 * np.log10(e25) # switches order with 25
        med[r]    = z50[ibest]
        yerr[r,0] = med[r] - z25[ibest]
        yerr[r,1] = z75[ibest] - med[r]
        #
        # select as best all those that are within the 25-75 percentile of the lowest RMSE
        #
        #print('ibest',ibest)
        worst_of_best = e75[ibest]
        #print('worst of best',worst_of_best)
        best = np.flatnonzero(e50 < worst_of_best )
        #print('good enough', best, e50[best])
        plot_results(rate,w,o,z25,z50,z75,best)
    plt.figure(figsize=(4,4))
    plt.errorbar(10*rates,med,yerr=yerr.T)
    plt.ylim(0,37)
    plt.grid(True)
    plt.xlabel('sampling rate (%)')
    plt.ylabel('PSNR (dB)')
    plt.title('Recovery quality vs. sampling rate')
    plt.savefig('recovery_vs_rate.svg')
    #
    # for fixed rate and (best) width and overlap
    # see variation between images
    best_width = 32
    best_overlap = 1.0/16 # ratio
    best_stride = 30 # 32 - 32*overlap_ratio = 32 - 32*(1/16)
    # columns are
    # 0 Rate (1 to 5 representing 10% to 50%)
    # 1 patch width
    # 2 STRIDE
    # 3 number of non overlapping samples in patch (effective degrees of freedom)
    # 4 stride
    # 5 compressed samples per patch
    # 7 Kodak image number
    # 8 random seed
    # 9 RMSE
    # 10 PSNR
    # select rate = 2
    kr = set(np.flatnonzero(S[:,0] == 2))
    # select width = 32
    kw = set(np.flatnonzero(S[:,1] == 32))
    # select stride 30
    ks = set(np.flatnonzero(S[:,2] == 30))
    krws = list(kr & kw & ks)
    #
    # summarize result for each image
    #
    images = np.unique(S[:,-4])
    seeds  = np.unique(S[:,-3])
    x = np.arange(len(images))
    y = np.zeros(len(images))
    boxdata = np.zeros((len(seeds),len(images)))
    for i,img in enumerate(images):
        ki = set(np.flatnonzero(S[:,-4] == img))
        k = set(krws)
        Si = S[list(k & ki),:]
        boxdata[:,i] = Si[:,-1]
    plt.close('all')
    plt.figure(1,figsize=(6,4))
    plt.boxplot(boxdata,bootstrap=10000,labels=['kodak'+str(int(l)) for l in images])
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction quality at width=32px, overlap=2px')
    plt.savefig('best_vs_image.svg')
    #
    # latex table v 1
    #
    for r in range(1,6):
        kr = set(np.flatnonzero(S[:, 0] == r))
        for img in images:
            ki = set(np.flatnonzero(S[:,-4] == img))
            k = set(krws)
            Si = S[list(kr & kw & ks & ki),:]
            psnr = Si[:,-1]
            #print(psnr)
            p25 = np.round(np.percentile(psnr,25),2)
            p50 = np.round(np.percentile(psnr,50),2)
            p75 = np.round(np.percentile(psnr,75),2)
            print(f'{r:6} &{img:6.0f} &{p25:6.2f} &{p50:6.2f} &{p75:6.2f} \\\\')

    #
    # latex table v2
    #
    for r in range(1,6):
        kr = set(np.flatnonzero(S[:, 0] == r))
        print(f'{r:6} ',end=' ')
        nimages = len(images)
        for img in images:
            ki = set(np.flatnonzero(S[:,-4] == img))
            k = set(krws)
            Si = S[list(kr & kw & ks & ki),:]
            psnr = Si[:,-1]
            p50 = np.round(np.percentile(psnr,50),2)
            print(f'&{p50:6.1f}',end=' ')
        print('\\\\')
