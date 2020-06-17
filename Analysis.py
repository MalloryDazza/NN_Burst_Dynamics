#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Maths
import numpy as np
from scipy.signal import argrelextrema as localext
from sklearn.cluster import DBSCAN
from scipy.signal import convolve
from scipy.interpolate import interp1d

#plot
import matplotlib.pyplot as plt
import itertools as it

#Shapely
from shapely.geometry import Point, MultiPoint, Polygon, LineString, MultiLineString
from shapely.prepared import prep

"""Paths and patches"""

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import asarray, concatenate, ones

class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface
    def __init__(self, context):
        if isinstance(context, dict):
            self.context = context['coordinates']
        else:
            self.context = context

    @property
    def exterior(self):
        return (getattr(self.context, 'exterior', None)
                or self.context[0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context[1:]
        return value

def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = concatenate([
        concatenate([asarray(t.exterior)[:, :2]] +
                    [asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = concatenate([
        concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPath(polygon), **kwargs)

def load_data(spike_file, tmin=-np.inf, tmax=np.inf, with_space = True):
    '''
    Load data into  NTXY lists
    
    Parameters:
    -----------
    - Spike_file: string, file where the activity is stored (column neuron id, 
                  (NEST) spike time, positions X, Y)
    - tstart: float, starting time of the analysis 
    - tstop: float , ending time of the analysis
    - with_space = bool, 
    Return:
    -------
            1 lists of 4 elements: neurons (NESST) id, spike time, 
            positions X, positions Y
    '''                 
    if with_space == True:
        return_list = []
        with open(spike_file, "r") as fileobject:
            for i, line in enumerate(fileobject):
                if not line.startswith('#'):
                    lst = line.rstrip('\n').split(' ')
                    # !!! NEST ids start at 1 !!! 
                    if float(lst[1]) > tmin and float(lst[1]) < tmax:
                        return_list.append([int(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
        NTXY = np.array(sorted(return_list, key = lambda x:x[1]))
        return_list = None
        
    else:
        return_list = []
        with open(spike_file, "r") as fileobject:
            for i, line in enumerate(fileobject):
                if not line.startswith('#'):
                    lst = line.rstrip('\n').split(' ')
                    # !!! NEST ids start at 1 !!! 
                    if float(lst[1]) > tmin and float(lst[1]) < tmax:
                        return_list.append([int(lst[0]),float(lst[1])])
        NTXY = np.array(sorted(return_list, key = lambda x:x[1]))
        return_list = None
        
    return NTXY

def load_raster(file_name, tmin=-np.inf, tmax=np.inf, with_space = True):
    '''
    Return raster as a list of size N_neurons x N_spikes
    '''
    if tmin > tmax:
        raise Exception('BadProperty tmin has to be smaller than tmax')
    return_list = []
    with open(file_name, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                if with_space == True:
                    # !!! NEST ids start at 1 !!! 
                    if float(lst[1]) > tmin and float(lst[1]) < tmax:
                        return_list.append([int(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
                else:
                    if float(lst[1]) > tmin and float(lst[1]) < tmax:
                        return_list.append([int(lst[0]),float(lst[1])])

    NTXY = np.array(sorted(return_list, key = lambda x:x[1]))
    senders = NTXY[:,0].astype(int)
    times   = NTXY[:,1]
    
    neurons, count = np.unique(senders, return_counts = True)
    neurons = np.sort(neurons)
    length = np.max(count)
    
    activity = np.zeros((len(neurons),length))
    
    if with_space == True:
        pos     = NTXY[...,2:]
        positions = []
    
    for i,nn in enumerate(neurons):
        msk = (senders == nn)
        tspk = times[msk]
        l = len(tspk)
        if l < length:
            tspk = np.append(tspk, [np.NaN]*(length-l))
        activity[i] = tspk
        if with_space == True:
            positions.append(pos[nspk[0]])
            
    if with_space == True:
        return activity, np.array(positions).astype(float)
        
    else:
        return activity



def mean_simps(x, y, x1, x2):
    '''
    Compute the mean of y between x1 and x2 with the simps method from
    scipy to compute the integral
    '''
    from scipy.integrate import simps
    
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    x = x[id1:id2]
    
    return simps(y,x) / (y[-1] - y[1])

def mean_direct(y, x, x1, x2):
    '''
    Compute the mean for evenly spaced x
    '''
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    
    return np.mean(y)

def normalise_array(array):
    return (array - np.min(array)) / ( np.max(array) - np.min(array))

def convolve_gauss(fr, sigma, dt, crop=5.):
    '''
    Convolve a spiketrain by an gaussian kernel with width `sigma`.

    Parameters
    ---------
    - fr : 1D array
        Firing rate.
    - sigma : double
        Timescale for the gaussian decay.
    - dt : double
        Timestep for the "continuous" time arrays that will be returned.
    - crop : double, optional (default: 5.)
        Crop the gaussian after `crop*sigma`.

    Returns
    -------
    ts, fr : 1D arrays
    '''
    # create the gaussian kernel
    tkernel  = np.arange(-crop*sigma, crop*sigma + dt, dt)
    ekernel  = np.exp(-(tkernel/(2*sigma))**2)
    ekernel /= np.sum(ekernel)
    # convolve
    fr = np.array(convolve(fr, ekernel, "same"))
    return fr

def all_time_neuron_phase(Activity_Raster, times):
    '''
    Compute the phase at times (1d array) for all neurons
    
            !!! times must not be too long !!!
            
    '''
    phases = np.zeros(shape = (len(Activity_Raster),len(times)))
    for i,r in enumerate(Activity_Raster):
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(times,r) - 1
        ph = (times - r[idx]) / isi[idx]
        idx = np.isnan(ph)
        ph[idx] = 0
        phases[i] = ph
        
    return phases

def single_time_neuron_phase(Activity_Raster, time, Kuramoto = True):
    '''
    Compute the phase of all neurons at 'time'
    
    Params :
    --------
    
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - time            : float
                       time at which we compute the phase
    - Kuramoto        : bool
                        Wether to return the phase in [-2 pi , 0] (True,
                        Default) or in [0,1] (False)
    Return :
    --------
            phase of the neurons phi = (t-t_k)/(t_k-t_k-1) as function of time
    '''
    phi = []
    for r in Activity_Raster:
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(time,r) - 1
        ph = (time - r[idx]) / isi[idx]
        if ph < 0:
            ph = 0.
        phi.append(ph)
    if Kuramoto:
        return np.array(phi)*2*np.pi
    else:
        return np.array(phi)

def all_time_network_phase(Activity_Raster, times, smooth = False,
                           after_spike_value=0.5):
    '''
    Compute the phase of all neurons at 'times'
    
    Params :
    --------
    
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - times            : 1d array
                       times at which we compute the phase
    - smooth        : bool
                        Wether to smooth the phase by gaussian convolution
    - after_spike_value : Float
                        Phase value to assign when a neurons stopped spiking
                        and before he starts
        return :
        --------
                phase of the neurons phi = (t-t_k)/(t_k-t_k-1) as function of time
    '''
    phases = np.zeros(shape = len(times))
    
    for r in Activity_Raster:
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(times,r) - 1
        ph = (times - r[idx]) / isi[idx]
        msk = (times < np.nanmin(r)) + (times >= np.nanmax(r))
        ph[msk] = after_spike_value
        phases += ph
    phases /= len(Activity_Raster)
    
    if smooth == True:
        dt = times[1]-times[0]
        phases = convolve_gauss(phasess, sigma = dt, dt = dt, crop = 2*dt )
    
    return np.array(phases)

def kuramoto_od(phases):
    '''
    Compute the Kuramoto order parameter
    
    use for one time step
    '''
    j = np.complex(0,1)
    S = sum(np.exp(j*np.array(phases))) / len(phases)
    return np.abs(S) , np.angle(S)

def kuramoto_radius(phases):
    '''
    Compute the Kuramoto order parameter
    
    Parameters :
    ------------
    - phases  : nd array shape = (n_neurons, n_times)
    
    Result : 
    -------- 
    Kuramoto as a function of time
    '''
    j = np.complex(0,1)
    S = np.sum(np.exp(j*np.array(phases)), axis = 0) / phases.shape[0]
    return np.abs(S)

def mean_simps(x, y, t1, t2):
    '''
    Compute the mean of y between x1 and x2 with the simps method from
    scipy to compute the integral
    '''
    from scipy.integrate import simps
    
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    x = x[id1:id2]
    
    return simps(y,x) / (y[-1] - y[1])

def mean_direct(y, x, x1, x2):
    '''
    Compute the mean for evenly spaced x
    '''
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    
    return np.mean(y)
    
def Burst_times(Activity_Raster, time_array, th_high = 0.6, 
                th_low = 0.4, ibi_th = 0.9, plot = False):
    '''
    Compute the starting and ending time of a burst
    
    Params :
    --------
    - Activity_Raster : ND array of 1D arrays
                       spike trains of all neurons
    - time_array      : 1D array
                        times for the analaysis
    - th_high         : float
                        upper threshold use to separate the network phase
    - th_low          : float
                        lower threshold use to separate the network phase
    - ibi_th          : float
                        threshold use to separate the detected local maxima
    - plot            : Bool
                        Wether to plot of not the results
    Return :
    --------
            2D array, starting and ending point of bursts
    '''
    def check_minmax(idx):
        array = ( Net_phi[idx] > th_high )
        ar = np.append([0],array)
        br = np.append(array,[1])
        return ar + br
    
    def is_there_one(array):
        if 0 in array or 2 in array:
            return True
        else:
            return False
    
    #Compute the network phase
    dt = time_array[1] - time_array[0]
    Net_phi = all_time_network_phase(Activity_Raster, time_array)
    if plot == True:
        f,a = plt.subplots()
        a.plot(time_array, Net_phi, 'k', alpha = .6)
    Net_phi = convolve_gauss(Net_phi, sigma = 2*dt, dt=dt, crop = 4.)
    
    #f,a = plt.subplots()
    #a.plot(time_array,Net_phi)
    
    #Compute argmin and argmax of the network phase
    mask_up = (Net_phi > th_high)
    ph_up = Net_phi[mask_up]
    max_idx = localext(ph_up, np.greater)[0]
    
    mask_dw = (Net_phi < th_low)
    ph_dw = Net_phi[mask_dw]
    min_idx = localext(ph_dw, np.less)[0]
    
    if len(ph_up) == 0 or len(ph_dw) == 0:
        print('No oscillations')
    else:
        #first extrema is a max
        ts = time_array[mask_up][max_idx[0]]
        pop = np.where(time_array[mask_dw][min_idx] < ts)[0]
        min_idx = np.delete(min_idx, pop)
        #last extrema is a min
        ts = time_array[mask_dw][min_idx[-1]]
        pop = np.where(time_array[mask_up][max_idx] > ts)[0]
        max_idx = np.delete(max_idx, pop)

        time_bursts = []
        
        #Clustering extrema
        border_idx = []
        #find borders of clusters of minima and maxima
        for times,idx in [
                           [time_array[mask_up], max_idx], 
                           [time_array[mask_dw], min_idx]
                         ]:
            #if idx[0] == min_idx[0]:
                #a.vlines(times[idx], [0.25], [0.75], 'r')
            imi = np.diff(times[idx])
            th = np.mean(imi)
            w = np.where(imi > th*ibi_th)[0] + 1
            border_idx.append(idx[w])
            #if idx[0] == min_idx[0]:
                #a.vlines(times[idx[w]], [0], [0.5], 'b')
        
        
        idx = np.where(time_array[mask_dw][border_idx[1]] 
                       > time_array[mask_up][border_idx[0][-1]])[0]
        #take only one last min
        if len(idx) > 1:
            border_idx[1] = np.delete(border_idx[1], idx[1:])
        #get rid of the last border if its a maximum
        else:
            idx = np.where(time_array[mask_up][border_idx[0]]
                                    > time_array[mask_dw][border_idx[1][-1]])[0]
            if len(idx) > 0:
                border_idx[0] = np.delete(border_idx[0], idx)
        
        
        idx = np.where(time_array[mask_up][border_idx[0]] 
                       < time_array[mask_dw][border_idx[1][0]])[0]
        #take only one first max
        if len(idx) > 1:
            border_idx[0] = np.delete(border_idx[0], idx[:-1])
        #get rid of first minimum
        else:
            idx = np.where(time_array[mask_dw][border_idx[1]]
                                    < time_array[mask_up][border_idx[0][0]])[0]
            if len(idx) > 0:
                border_idx[0] = np.delete(border_idx[0], idx)
                
        #a.vlines(time_array[mask_dw][border_idx[1]], [0.75], [1.], 'r')
        
        #return only one extrema in each clusters
        for times,phi,idx,func,border in [
                           [time_array[mask_up], ph_up, max_idx, np.argmax, border_idx[0]], 
                           [time_array[mask_dw], ph_dw, min_idx, np.argmin, border_idx[1]] 
                         ]:
            
            loop = [0]
            loop.extend(border)
            colors = [plt.cm.viridis(each) for each in np.linspace(0.,1.,len(loop))]
            
            #if idx[0] == min_idx[0]:
                #a.vlines( times[loop], [0.75], [1.], 'k')
            
            for i in range(1,len(loop)):
                ts = loop[i-1]
                te = loop[i]
                
                mask = (idx < te) & (idx >= ts)
                #keep only the largest maxima and smallest minima
                grped_idx = idx[mask]
                gold_idx = func(phi[grped_idx])
                time_bursts.append(times[grped_idx][gold_idx])
                
                #if idx[0] == min_idx[0] and i == 1:
                    #a.vlines(times[grped_idx], [0.5], [.8], 'k')
                    #a.vlines(times[grped_idx][gold_idx], [0.8], [1.], 'grey')
        plt.show()
        time_bursts.sort()
        idx_bursts = [np.where(time_array == tb )[0][0] for tb in time_bursts]
        added_idx = check_minmax(idx_bursts)

        #sort min max alternating to remove double detected max or double detected min
        PROBLEM = is_there_one(added_idx)
        while PROBLEM:

            idx = np.where(added_idx[1:-1] == 0)[0]
            if len(idx) != 0:
                idx = idx[0]
                if Net_phi[idx_bursts[idx]] >= Net_phi[idx_bursts[idx+1]]:
                    idx_bursts.pop(idx)
                else:
                    idx_bursts.pop(idx+1)

            idx = np.where(added_idx[1:-1] == 2)[0]
            if len(idx) != 0:

                idx = idx[0]
                if Net_phi[idx_bursts[idx]] <= Net_phi[idx_bursts[idx+1]]:
                    idx_bursts.pop(idx)
                else:
                    idx_bursts.pop(idx+1)

            added_idx = check_minmax(idx_bursts)
            if added_idx[0] == 0:
                idx_bursts = idx_bursts[1:]
            added_idx = check_minmax(idx_bursts)
            if added_idx[-1] == 2:
                idx_bursts = idx_bursts[:-1]
            added_idx = check_minmax(idx_bursts)
            PROBLEM = is_there_one(added_idx)

        time_bursts = [time_array[i] for i in idx_bursts]

        if len(time_bursts) != 0:
            N_burst = int(len(time_bursts) / 2 )
            ibi = np.mean(np.diff(time_bursts)[range(1,2*N_burst-1,2)])
            if time_bursts[-1] > time_array[-1]-ibi:
                time_bursts.pop(-1)
                time_bursts.pop(-1)
            time_bursts = [[time_bursts[i],time_bursts[i+1]] 
                                for i in range(0,len(time_bursts)-1,2)]
            idx_bursts = [[idx_bursts[i],idx_bursts[i+1]] 
                                for i in range(0,len(time_bursts)-1,2)]

            if plot == True:
                a.plot(time_array, Net_phi)
                a.vlines(time_bursts, [0]*len(time_bursts), [1]*len(time_bursts), ['r','g'])
                plt.plot()

        else:
            print('No burst detected')
            if plot == True:
                f,a = plt.subplots()
                a.plot(time_array, Net_phi)
                plt.plot()
    
        return np.array(time_bursts), Net_phi
    
def First_To_Fire(Activity_Raster, time_end, time_start):
    '''
    Find neurons that fire before burst
    
    Params : 
    --------
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - time_end         : float 
                        time of the end of the previous burst
    - time_start       : float
                        time of the start of the burst to find ftf
    return:
    -------
            List of index of neuron that fire
    '''
    ret = []
    
    for i, spikes in enumerate(Activity_Raster):
        idxinf = np.where(spikes > np.mean([time_end, time_start]))[0]
        ftf    = np.where(spikes[idxinf] < time_start)[0]
        if len(ftf) != 0:
            ret.append(i)
    return ret
    
def first_to_fire_probability(spike_file, time_bursts, first_spike, threshold):
    '''
    Compute the probability for neurons to be class I or II first to fire
    
    Parameters:
    -----------
    spike_file: string, file to get the activity
    time_bursts: 1d array of float starting time of the bursts (to use as reference)
    first_spike: 1d array ; first spike of the burst (or time near it) 
    threshold: float ; time reference after the burst starts for class II neurons
    
    Return:
    -------
        Prob1, Prob2 as 2 Nd array with the probability for each neurons
        
    Note: Class I is from first spike to time_burts-1, class II is 2 ms around tim_bursts
    '''
    
    return_list = []
    with open(spike_file, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                return_list.append([int(lst[0]),float(lst[1])])
            
    NT      = np.array(sorted(return_list, key = lambda x:x[1]))
    neurons = set(NT[:,0])
    Prob1, Prob2 = np.zeros(len(neurons)), np.zeros(len(neurons))
    neurons = np.arange(1,len(neurons)+1)
    
    fspk = []
    for bstart,tb in zip(first_spike, time_bursts):
        spk_count = []
        #take the first spike of each neurons
        for nrn in neurons:
            msk = NT[:,0] == nrn
            i = np.where(NT[:,1][msk]>bstart)[0][0]
            spk_count.append(NT[:,1][msk][i] - tb)
        spk_count = np.array(spk_count)
        
        mask = (spk_count < -1)
        Prob1[mask] += 1.
        
        mask = (spk_count < threshold) & ~mask
        Prob2[mask] += 1.
    
    return Prob1/float(len(time_bursts)), Prob2/float(len(time_bursts))

def first_spk_cum_activity(spike_file, time_bursts, first_spike, x, bin_size):
    '''
    Compute the cumulative activity (with an histogram) as function of time,
    with linear extrapolation in between points.
    
    Parameters:
    -----------
    spike_file: string, file to get the activity
    time_bursts: 1d array of float starting time of the bursts (to use as reference)
    first_spike: 1d array, first spike of each burst (or time near and before it))
    x: abscisse points for linear approx
    bin_size: histogram bin size
    
    return :
    -------- 
            1d array 
    '''
    return_list = []
    with open(spike_file, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                return_list.append([int(lst[0]),float(lst[1])])
            
    NT = np.array(sorted(return_list, key = lambda x:x[1]))
    
    neurons, count = np.unique(NT[:,0], return_counts = True)
    norm    = float(len(neurons))
    length = np.max(count)
    
    activity = np.zeros((int(norm),length))
    for nrn in neurons:
        msk = NT[:,0] == nrn
        spk = NT[:,1][msk]
        l = len(spk)
        if l < length:
            spk = np.append(spk, [np.NaN]*(length-l))
        activity[int(nrn-1)] = spk
        
    fspk = []
    for bstart,tb in zip(first_spike, time_bursts):
        spk_count = activity - bstart
        mask_count = np.ma.masked_less_equal(spk_count, 0)
        spk_count = np.nanmin(mask_count, axis = 1) - tb + bstart
        print(spk_count[:5])
        bins = np.arange(np.min(spk_count)-1, np.max(spk_count)+1, bin_size)
        h,b = np.histogram(spk_count, bins = bins)
        b = b[:-1] + (b[1]-b[0])/2.
        h = h.cumsum() / norm
        f = interp1d(b, h)
        y = [0.]*np.sum((x<b[0]))
        msk = (x > b[0]) & (x < b[-1])
        y.extend(list(f(x[msk])))
        y.extend([1.]*(len(x)-len(y)))
        fspk.append( y )
        
    return np.array(fspk)

def closest_spk_cum_activity(spike_file, time_ref, x, bin_size, 
                            tmin = 0., tmax = np.inf):
    '''
    Compute the cumulative activity (with an histogram) as function of time,
    with linear extrapolation in between points of the cloest points of 
    specific reference.
    
    Parameters:
    -----------
    spike_file: string, file to get the activity
    time_ref: 1d array of float of times to use as reference
    x: abscisse points for linear approx
    bin_size: histogram bin size
    tmin, tmax : floats , time boundaries to load less data
    
    return :
    -------- 
            1d array 
    '''
    '''
    NT = load_data(spike_file, tmin, tmax, with_space = False)
    
    neurons, count = np.unique(NT[:,0], return_counts = True)
    norm           = float(len(neurons))
    length = np.max(count)
    
    activity = np.zeros((int(norm),length))
    
    for i,nrn in enumerate(neurons):
        msk = NT[:,0] == nrn
        spk = NT[:,1][msk]
        l = len(spk)
        if l < length:
            spk = np.append(spk, [np.NaN]*(length-l))
        activity[i] = spk
    '''
    
    activity = load_raster(spike_file, tmin=tmin,
                           tmax=tmax, with_space = False)
    norm           = float(len(activity))
    fspk = []
    
    for tr in time_ref:
        spk_count = activity - tr
        idx = np.nanargmin(np.abs(spk_count), axis = 1)
        spk_count = spk_count[...,idx].diagonal()
        b = np.arange(np.min(spk_count)-1, np.max(spk_count)+1, bin_size)
        h,b = np.histogram(spk_count, bins = b)
        b = b[:-1] + (b[1]-b[0])/2.
        h = h.cumsum() / norm
        f = interp1d(b, h)
        y = [0.]*np.sum((x<b[0]))
        msk = (x > b[0]) & (x < b[-1])
        y.extend(list(f(x[msk])))
        y.extend([1.]*(len(x)-len(y)))
        fspk.append( y )

    return np.array(fspk)

def probability_distribution(cumulative_distribution, x):
    '''
    Compute the probability distribution from the cumulative as a 
    differential
    '''
    dx = np.diff(x)
    x = x[:-1] + dx/2.
    
    dc = np.diff(cumulative_distribution)
    
    dc /= dx 
    
    return x, dc/np.sum(dc)
    
def average_curve( data_x, data_y, IC , FC, x ):
    '''
    Compute the average curve given by different realisation
    
    Parameters:
    -----------
    data_x : array N_points X N_realisation , x coordiantes of 
    every realisation of the measured curve
    daya_y : array N_points X N_realisation , corresponding y coordinates
    IC : tuple, values of x and y for the initial condition
    FC : tuple, values of x and y for the final condition
    x : 1d array, abscissa values where the curve will be computed
    '''
    
    y = np.zeros(x.shape)
    
    for s,e in zip(data_x,data_y):
        S = [IC[0]]
        S.extend(s)
        S.append(FC[0])
        
        E = [IC[1]]
        E.extend(e)
        E.append(FC[1])
        
        f = interp1d(S, E, fill_value = [np.NaN])
        
        y +=  f(x)
    return y / float(len(data_x))
