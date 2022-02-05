from wavescapes import *
from glob import glob
import numpy as np
import math
from scipy import ndimage
from scipy.stats import entropy
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, StandardScaler


MOZART_PROFILES = {
 'major': {0: 0.20033700035703508,
  1: 0.010812613711830977,
  2: 0.11399209672667372,
  3: 0.012104110819714938,
  4: 0.13638736763981654,
  5: 0.1226311293358654,
  6: 0.018993520966402003,
  7: 0.20490464831336042,
  8: 0.014611863068643751,
  9: 0.07414111247856302,
  10: 0.011351150687477073,
  11: 0.07973338589461738},
 'minor': {0: 0.1889699424221723,
  1: 0.008978237532936468,
  2: 0.11060533806574259,
  3: 0.1308781295283801,
  4: 0.011630786793101157,
  5: 0.11019324778803881,
  6: 0.029590747199631288,
  7: 0.21962043820162988,
  8: 0.07742468998919953,
  9: 0.012129908077215472,
  10: 0.020386372564054407,
  11: 0.07959216183789804}}


def most_resonant(utm):
    ''' Computes 3 NxNx1 matrices containing:
        the inverse entropy of the 6 coefficients at each point of the matrix
        the maximum value among the 6 coefficients
        the max coefficient'''
    utm_magnitude = np.abs(utm)
    utm_max = np.max(utm_magnitude[:,:,1:], axis=2)
    utm_argmax = np.argmax(utm_magnitude[:,:,1:], axis=2)

    utm_entropy = MinMaxScaler().fit_transform(1 - (entropy(utm_magnitude[:,:,1:], axis=2) / entropy(np.array([1,1,1,1,1,1]))))  #the normalization is completely arbitrary and should be changed

    return (utm_max, utm_entropy, utm_argmax)

def custom_utm_to_ws_utm(utm_custom, utm_argmax, utm, how='max'):
    """
    Converts an upper triangle matrix filled with Fourier coefficients into 
    an upper triangle matrix filled with color values that serves as the mathematical model
    holding the color information needed to build the wavescapes plot.
    
    """    
    shape_x, shape_y = np.shape(utm)[:2]
    #RGB => 3 values, RGBA => RGB + 1 value, raw values => angle & magnitude => 2 values
    channel_nbr = 4
    default_value = 0.0
    default_type = np.uint64 
    #+1 to differentiate empty elements from white elements later down the line.
    res = np.full((shape_x, shape_y, channel_nbr), default_value, default_type)
    
    for y in range(shape_y):
        for x in range(shape_x):
            if np.any(utm[y][x]):
                angle = utm_argmax[y][x]
                if how == 'max':
                    magn = zeroth_coeff_norm(utm[y][x], utm_custom[y][x])
                else:
                    magn = utm_custom[y][x]
                    if magn > 1:
                        magn = 1
                    
                res[y][x] = circular_hue_revised(angle, magnitude=magn) 
    return res

def zeroth_coeff_norm(value, curr_max):
    zero_c = value[0].real
    if zero_c == 0.:
        return (0.,0.)#([0xff]*3
    magn = curr_max/zero_c
    return magn

    
def center_of_mass(index, utm):
    shape_x, shape_y = np.shape(utm)[:2]
    for y in range(shape_y):
        for x in range(shape_x):
            if np.any(utm[y][x]):
                utm[y][x] = zeroth_coeff_norm(utm[y][x], utm[y][x])
    utm_interest = np.abs(utm[:,:,index])
    vcom, hcom = ndimage.measurements.center_of_mass(utm_interest)
    return (hcom/shape_x, vcom/shape_y)

def pitch_class_matrix_to_tritone(pc_mat, build_utm = True):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    """
    pcv_nmb, pc_nmb = np.shape(pc_mat)
    #+1 to hold room for the 0th coefficient
    coeff_nmb = int(pc_nmb/2)+1
    res_dimensions = (pcv_nmb, coeff_nmb)
    res = np.full(res_dimensions, (0.), np.float64)

    for i in range(pcv_nmb): 
        res[i] = (pc_mat[i] * np.roll(pc_mat[i], 6))[:coeff_nmb] #coeff 7 to 11 are uninteresting (conjugates of coeff 6 to 1).
    
    if build_utm:
        new_res = np.full((pcv_nmb, pcv_nmb, coeff_nmb), (0.), np.float64)
        new_res[0] = res 
        res = build_custom_utm_from_one_row(new_res) #have to recompute from zero but normalizing norm2 of the vector
        
    res = np.mean(res, axis=2)
    #res[res != 0.] = 1 # for now boolean value
    return res.reshape((pcv_nmb, pcv_nmb, 1))


def pitch_class_matrix_to_minor_major(pc_mat, rotated_kp, build_utm = True):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    """
    pcv_nmb, _ = np.shape(pc_mat)
    res_dimensions = (pcv_nmb, 2)
    res = np.full(res_dimensions, (0.), np.float64)

    for i in range(pcv_nmb):
        res[i] = np.array(max_correlation(pc_mat[i], rotated_kp))#coeff 7 to 11 are uninteresting (conjugates of coeff 6 to 1).
    
    if build_utm:
        new_res = np.full((pcv_nmb, pcv_nmb, 2), (0.), np.float64)
        new_res[0] = res 
        res = build_custom_utm_from_one_row(new_res) #this does not make sense for major and minor: keep all rotations
        
    return res

def max_correlation(row, rotated_kp):
    coeffs_major = np.array([pearsonr(row, kp)[0] for kp in rotated_kp.values()])[:12]
    coeffs_minor = np.array([pearsonr(row, kp)[0] for kp in rotated_kp.values()])[12:]
    return coeffs_major.max(), coeffs_minor.max()

def build_utm_from_one_row(res):
    """
    given a NxN matrix whose first row is the only
    one that's filled with values, this function fills
    all the above row by summing for each row's element
    the two closest element from the row below. This
    method of summing builds an upper-triangle-matrix
    whose structure represent all hierarchical level.
    """
    pcv_nmb = np.shape(res)[0]
    for i in range(1, pcv_nmb):
        for j in range(0, pcv_nmb-i):
            res[i][i+j] = res[0][i+j] + res[i-1][i+j-1]
    return res

def build_custom_utm_from_one_row(res, how='mean'):
    """
    """
    pcv_nmb = np.shape(res)[0]
    for i in range(1, pcv_nmb):
        for j in range(0, pcv_nmb-i):
            if how == 'mean':
                res[i][i+j] = np.mean(np.array([res[0][i+j],res[i-1][i+j-1]]), axis=0)
            else:
                res[i][i+j] = np.max(np.array([res[0][i+j],res[i-1][i+j-1]]), axis=0)
    return res




## functions as they are in Cedrics code

def stand(v):
    """Convert value to hex"""
    v = np.round(v, 5) # to remove
    assert v <= 1, f"Value cannot exceed 1 but is {v}"
    return int(v*0xff)

def two_pi_modulo(value):
    value = value*2*math.pi/6 #addition to work with 6 coefficients colouring
    return np.mod(value, 2*math.pi)
    
def step_function_quarter_pi_activation(lo_bound, hi_bound, value):
    #in the increasing path branch
    if value >= lo_bound and value <= lo_bound + math.pi/3:
        return ((value-lo_bound)/(math.pi/3))
    #in the decreasing path branch
    elif value >= hi_bound and value <= hi_bound + math.pi/3:
        return 1-((value-hi_bound)/(math.pi/3))
    else:
        #the case of red 
        if lo_bound > hi_bound:
            return 0 if value > hi_bound and value < lo_bound else 1
        else:
            return 1 if value > lo_bound and value < hi_bound else 0
    
def circular_hue_revised(angle, magnitude=1.):
    # np.angle returns value in the range of [-pi : pi], where the circular hue is defined for
    # values in range [0 : 2pi]. Rather than shifting by a pi, the solution is for the negative
    # part to be mapped to the [pi: 2pi] range which can be achieved by a modulo operation.
            
    #Need to shift the value with one pi as the range of the angle given is between pi and minus pi
    #and the formulat I use goes from 0 to 2pi.
    angle = two_pi_modulo(angle)
    green = lambda a: step_function_quarter_pi_activation(0, math.pi, a)
    blue = lambda a: step_function_quarter_pi_activation(math.pi*2/3, math.pi*5/3, a)
    red = lambda a: step_function_quarter_pi_activation(math.pi*4/3, math.pi/3, a)
    value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)), stand(magnitude))
    
    return value



# still to adjust from Digital Musicology project

def compute_magnitude_entropy(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

def compute_magnitudes(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    return sel[4]

def compute_peaks(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = [ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec]
    return [len(find_peaks(list(magnitudes))[0]) for magnitudes in sel]

def compute_entropy_phase(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.round(np.angle(apply_dft_to_pitch_class_matrix(arr1)), 2)
    vec = []
    coeffs = []
    height = int(utm.shape[0] * ver_ratio)-1
    sel = np.array(utm[height,:,:]).T
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

