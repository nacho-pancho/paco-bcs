import numpy as np
from PIL import Image as img
from PIL import ImageChops as imgop
from PIL import ImageMorph

morph_lb = ImageMorph.LutBuilder(op_name='dilation4')
morph_dil4 = ImageMorph.MorphOp(morph_lb.build_lut())
morph_lb = ImageMorph.LutBuilder(op_name='dilation8')
morph_dil8 = ImageMorph.MorphOp(morph_lb.build_lut())


def grid_size(l, w, s):
    return int(np.ceil(float(l-w)/float(s))) + 1
        
def padded_size(l, w, s):    
    # ultimo patch empieza en (n-1)*s
    # senial tiene que llegar a (g-1)*s + w = np >= n
    # (g-1)*s +w >= n
    # s*(g-1) >= n-w
    # g-1 >= (n-w)/s
    # g >= (n-w)/s + 1
    g = grid_size(l,w,s)
    return (g-1)*s + w


def pad_image(img, w1, w2, s1, s2):
    '''
    Enlarge image so that it is exactly covered by the tiling defined by
    the patch width w and stride s
    '''
    d0,d1 = img.shape
    dd0 = padded_size(d0,w,s)
    dd1 = padded_size(d1,w,s)
    padded_img = np.zeros((dd0,dd1))
    padded_img[ :d0, :d1 ] = img
    # mirror is best for DCT
    if dd0 > d0:
        pad0 = dd0-d0
        print(('pad along dim 0',pad0))
        padded_img[ d0:, :d1 ] = img[ (d0-1):(d0-1-pad0):-1, : ]
    if dd1 > d1:
        pad1 = dd1-d1
        print(('pad along dim 1',pad1))
        padded_img[ :d0, d1: ] = img[ :, (d1-1):(d1-1-pad1):-1 ]
    if dd0 > d0 and dd1 > d1:
        padded_img[ d0:, d1: ] = img[ (d0-1):(d0-1-pad0):-1, (d1-1):(d1-1-pad1):-1 ]
    return padded_img

#
# given a list of signal bands (e.g., R, G, B) with integer elements
# and a bpp per band (e.g. 8 for 8 bit images)
# returns a single-band signal where each sample contains the interleaved bits
# of each band, MSB first, in the order they appear in the list.
# for example, if signal_bands = (G, B, R) (green is more important)
# and the bits of G[0] are g0,g1,g2,...,g7, those of R[] are r0,r1, r2...
# then the output sample bits are the 24 bit samples S[0] = [g0,r0,b0,g1,r1,b1,...,g7,r7,b7]
#
def interleave_bands(signal_bands):
    nbands = len(signal_bands)
    output_signal_shape = signal_bands[0].shape
    # first implementation: assume unsigned 8 bit samples
    input_bpp = 8
    output_signal = np.zeros(output_signal_shape,dtype=np.uint32)
    bandits = list()
    for i in range(nbands):
        bandits.append( np.nditer( signal_bands[i]) )
    outit = np.nditer( output_signal, op_flags=['writeonly'] )
    k = 0
    while not outit.finished:      
        x = 0
        omask = 1
        imask = 1
        for j in range(input_bpp):
            for i in range(nbands-1,-1,-1):
                if bandits[i].value & imask:
                    x = x | omask
                omask = omask <<  1
            imask = imask << 1
        outit.value[...] = x
        for i in range(nbands):
            bandits[i].iternext()
        outit.iternext()
        k = k + 1
    return output_signal


def deinterleave_bands(interleaved_signal,nbands):
    output_band_shape = interleaved_signal.shape
    # first implementation: assume unsigned 8 bit samples
    output_bpp = 8
    inputit = np.nditer( interleaved_signal, op_flags=['readonly'] )    
    output_signal_bands = list()
    bandits = list()
    for i in range(nbands):
        band_i = np.zeros(output_band_shape,dtype=np.uint8)
        output_signal_bands.append(band_i)
        bandits.append( np.nditer( band_i, op_flags=['writeonly']) )
    while not inputit.finished:      
        x = inputit.value
        omask = 1
        imask = 1
        # give one bit to each band
        for j in range(output_bpp):
            # last band is least significant, that is why we go backwards
            for i in range(nbands-1,-1,-1):
                if x & imask:
                    bi = bandits[i].value
                    bandits[i].value[...] = bi | omask
                imask = imask << 1
            omask = omask <<  1
        for i in range(nbands):
            bandits[i].iternext()
        inputit.iternext()
    return output_signal_bands

def dictionary_mosaic(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w))
            im[i0:i1,j0:j1] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im

def dictionary_mosaic_color(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m/3 ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng,3))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w,3))
            im[i0:i1,j0:j1,:] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im
