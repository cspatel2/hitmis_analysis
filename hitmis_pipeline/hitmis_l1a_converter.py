# %% Imports
from __future__ import annotations
import os
import sys
import glob
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr
from skimage import transform
import lzma
import pickle
import datetime
import argparse
from skmpython import TransformImage
# %% Paths
PATH = os.path.dirname(os.path.realpath(__file__))
# %% Argument Parser
parser = argparse.ArgumentParser(
    description='Convert HiT&MIS L0 data to L1 data, with exposure normalization and timestamp regularization (1 image for every 4 minutes), which separates data by filter region and performs line straightening using ROI listed in accompanying "hitmis_roi.csv" file, and edge lines in "edge_detection" directory. The program will not work without these files present.')
# %% Add arguments
parser.add_argument('rootdir',
                    metavar='rootdir',
                    type=str,
                    help='Root directory containing HiT&MIS data')
parser.add_argument('dest',
                    nargs='?',
                    default=os.getcwd(),
                    help='Root directory where L1 data will be stored')
parser.add_argument('--flatfield',
                    required=False,
                    type=str,
                    help='Specify directory for flat field files.')
parser.add_argument('--noflatfield',
                    required=False,
                    action='store_true',
                    help='Do not apply flat field correction.')
parser.add_argument('--wl',
                    required=False,
                    type=float,
                    help='Wavelength to process')
parser.add_argument('--prefix',
                    required=False,
                    type=str,
                    help="Files will be named `<prefix>_YYYYMMDD_WL.nc")
parser.add_argument('--align',
                    required=False,
                    type=str,
                    help="Specify the path to the alignment file.")
parser.add_argument('--force', '-f',
                    required=False,
                    action='store_true',
                    help='Force overwrite of existing files.')

# %% Get all subdirs


def list_all_dirs(root):
    flist = os.listdir(root)
    # print(flist)
    out = []
    subdirFound = False
    for f in flist:
        if os.path.isdir(root + '/' + f):
            subdirFound = True
            out += list_all_dirs(root + '/' + f)
    if not subdirFound:
        out.append(root)
    return out


def getctime(fname):
    words = fname.rstrip('.fit').split('_')
    return int(words[-1])


# %% Define resolution function
resolutions = {}
ifile = open(os.path.join(PATH, 'hitmis_resolution.txt'))
for line in ifile:
    words = line.rstrip('\n').split()
    resolutions[float(words[0])] = [float(w) for w in words[1:]]
ifile.close()

# %% Define ROI function
roidict = {}


def get_roi(wl, cond=False):
    global roidict
    if int(wl*10) not in roidict.keys():
        roi = np.loadtxt(os.path.join(PATH, 'hitmis_roi_eclipse_hms1.csv'), skiprows=1,
                         delimiter=',').transpose()
        coord = np.where(roi == wl)
        coords = roi[2:, coord[1]]
        try:
            xmin = int(coords[0]) * (2 if cond else 1)
            xmax = int(coords[0] + coords[2]) * (2 if cond else 1)
            ymin = int(coords[1]) * (2 if cond else 1)
            ymax = int(coords[1] + coords[3]) * (2 if cond else 1)
        except Exception:
            return {'xmin': -1, 'ymin': -1, 'xmax': -1, 'ymax': -1}
        roidict[int(wl*10)] = {'xmin': xmin,
                               'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    else:
        return roidict[int(wl*10)]
# %% Defines fit lines


def get_lines(wl):
    roi = get_roi(wl)
    efile = 'edge_detection/%d_edge.txt' % (wl*10)
    data = np.loadtxt(os.path.join(PATH, efile)).transpose()
    # data[0] -= roi['xmin']
    # data[1] -= roi['ymin']
    return data


# %% Wavelengths
# wls = [777.4, 427.8,486.1, 630.0, 557.7, 656.3 ]
wls = [486.1, 630.0, 557.7, 656.3 ]

# %% Line straightening functions
pcoeffdict = {}
projdict = {}
coladjdict = {}
transformptsdict = {}
flatfielddict = {}


def get_wl(file: str | list):
    if isinstance(file, str):
        file = file.rsplit('.', 1)[0]
        wl = file.rsplit('_', 1)[-1]
        return int(wl)
    return list(map(get_wl, file))


def flat_field(wl, img, flatdir=None):
    global flatfielddict
    key = int(wl*10)
    if key not in flatfielddict.keys():
        files = glob.glob(flatdir)
        keys = get_wl(files)
        for idx, _key in enumerate(keys):
            with xr.open_dataset(files[idx]) as ds:
                flatfielddict[_key] = ds['avg'].values[::-1, ::-1]

    img /= flatfielddict[key]
    return img

def flat_field_img(wl, flatdir=None):
    global flatfielddict
    key = int(wl*10)
    if key not in flatfielddict.keys():
        files = glob.glob(flatdir)
        keys = get_wl(files)
        for idx, _key in enumerate(keys):
            with xr.open_dataset(files[idx]) as ds:
                flatfielddict[_key] = ds['avg'].values
    return flatfielddict[key]

def transform_gen(points, wl):
    """Transform generator function

    Args:
        points (np array): Array of points on the output image (col, row)
        fitpoly (np array): Polynomial coefficients
        col_adj (float): Column adjustment value
        deg (int): Degree of the fit polynomial.

    Returns:
        [type]: [description]
    """
    global pcoeffdict, coladjdict, transformptsdict
    if int(wl*10) not in pcoeffdict.keys():
        raise RuntimeError('poly coefficients do not exist')
    if int(wl*10) not in transformptsdict.keys():
        fitpoly = pcoeffdict[int(wl*10)]
        col_adj = coladjdict[int(wl*10)]
        for i in range(points.shape[0]):
            coord = points[i]
            x = 0
            for i in range(len(fitpoly)):
                x += fitpoly[i]*coord[1]**(len(fitpoly) - 1 - i)
            coord[0] -= col_adj - x
        transformptsdict[int(wl*10)] = points
    else:
        points = transformptsdict[int(wl*10)]

    return points


def straighten_image(img, wl, flatdir, deg=2, do_flatfield=True):
    """Straighten HiT&MIS image for given segment

    Args:
        img ([type]): [description]
        wl ([type]): [description]
        deg (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    global pcoeffdict
    roi = get_roi(wl)  # get ROI
    cimg = img[roi['ymin']:roi['ymax'],
               roi['xmin']:roi['xmax']]  # cropped image
    # if do_flatfield:
    #     cimg = flat_field(wl, cimg, flatdir)
    if int(wl*10) not in pcoeffdict.keys():
        poi = get_lines(wl)
        proj = poi.copy()
        proj[0, :] = proj[0, :].max()
        col_adj = proj[0, :].max()
        pcoeff = np.polyfit(get_lines(wl)[1], get_lines(wl)[
                            0], deg)  # x for given y
        col_max = 0
        for i in range(len(pcoeff)):
            col_max += pcoeff[i]*roi['ymax']**(deg - i)
        pcoeffdict[int(wl*10)] = pcoeff
        projdict[int(wl*10)] = proj
        coladjdict[int(wl*10)] = col_adj
    else:
        proj = projdict[int(wl*10)]
    wimg = transform.warp(cimg, transform_gen, map_args={
                          'wl': wl}, mode='constant', cval=np.nan)
    return (wimg, proj)

# %% Convert time delta to HH:MM:SS string


def tdelta_to_hms(tdelta):
    if tdelta < 0:
        return 'Negative time delta invalid'
    tdelta = int(tdelta)
    tdelta_h = tdelta // 3600
    tdelta -= tdelta_h * 3600
    tdelta_m = tdelta // 60
    tdelta -= tdelta_m * 60
    outstr = ''
    if tdelta_h > 0:
        outstr += str(tdelta_h) + ' h '
    if tdelta_m > 0:
        outstr += str(tdelta_m) + ' m '
    outstr += str(tdelta) + ' s'
    return outstr
# %% Save straightened images, no other processing


def get_imgs_from_files(flist, wl, align_file: str, flatdir: str, do_flatfield: bool):
    num_frames = len(flist)
    otstamp = None
    tconsume = 0
    imgdata = []
    expdata = []
    tdata = []

    with lzma.open(os.path.join(PATH, 'pixis_dark_bias.xz'), 'rb') as dfile:
        dark_data = pickle.load(dfile)

    pbar = tqdm(range(num_frames))
    pbar.set_description(f'Wavelength: {wl} nm')
    for i in pbar:
        fname = flist[i]
        try:
            _fimg = pf.open(fname)
        except Exception as e:
            print(f'Exception {fname}: {e}')
            continue
        fimg = _fimg[1]
        try:
            data = np.asarray(fimg.data, dtype=float)
            exposure = fimg.header['exposure_ms']*0.001
            data -= dark_data['bias'] + (dark_data['dark'] * exposure)
            # apply new routine
            # 1. load image
            ndata = TransformImage(data)
            # 2. load transformations
            ndata.load_transforms(align_file)
            # 3. downsample to source sampling
            ndata.downsample()
            # 4. get the corrected field
            data = ndata.data
            # 5. apply the flat field
            # data -= np.average(data[950:, 100:])
            data = straighten_image(
                data, wl, flatdir, do_flatfield=do_flatfield)[0]
        except Exception as e:
            print(f'Exception {fname}: {e}')
            continue

        tstamp = (fimg.header['timestamp']*0.001)
        imgdata.append(data)
        tdata.append(tstamp)
        expdata.append(exposure)

    return (imgdata, expdata, tdata)


# %% Parse arguments
args = parser.parse_args()

rootdir = args.rootdir
wl = args.wl
flatdir = args.flatfield

if flatdir is None:
    flatdir = os.path.join(PATH, 'hms1_flat_fields')

if not os.path.exists(flatdir) or os.path.isfile(flatdir):
    print(f'Error: {flatdir} does not exist or is not a directory.')
    sys.exit(0)

flatdir = os.path.join(flatdir, '*.nc')

if wl is not None:
    wls = [wl]

fprefix = args.prefix
if fprefix is None:
    fprefix = 'hitmis_resamp'

align_file = args.align
if align_file is None:
    align_file = os.path.join(PATH, 'custom_alignment_hms1_eclipseday.txt')

if not os.path.exists(align_file):
    print(f'Specified alignment file {align_file} does not exist.')
    sys.exit()

if os.path.isdir(align_file):
    print(f'Alignment file: {align_file} is a directory.')
    sys.exit()

if not os.path.isdir(rootdir):
    print('Specified root directory for L0 data does not exist.')
    sys.exit()

destdir = args.dest

do_flatfield = not args.noflatfield

overwrite = args.force

if not do_flatfield:
    print('Flat field correction disabled.')

print('Files are being stored in:', end='\n\t')
print(destdir, end='\n\n')
# if destdir is None or not os.path.isdir(destdir):
#     print('Specified destination directory does not exist, output L1 data will be stored in current directory.\n')
#     destdir = './'

# %% Load in file list
dirlist = list_all_dirs(rootdir)
# %% Get all files

filelist = []
for d in dirlist:
    if d is not None:
        f = glob.glob(d+'/*.fit')
        f.sort(key=getctime)
        filelist.append(f)

flat_filelist = []
for f in filelist:
    for img in f:
        flat_filelist.append(img)

flist = flat_filelist
flist.sort(key=getctime)
# %% Get timeframe
start_date = datetime.datetime.fromtimestamp(getctime(flist[0])*0.001)
end_date = datetime.datetime.fromtimestamp(getctime(flist[-1])*0.001)
print('First image:', start_date)
print('Last image:', end_date)
print('\n')
# %% Break up into individual days, day is noon to noon
st_date = start_date.date() - datetime.timedelta(days=1)
lst_date = end_date.date() + datetime.timedelta(days=1)
main_flist = {}
all_files = []
print('Dates with data: ', end='')
data_found = False
first = True
while st_date <= lst_date:
    _st_date = st_date
    start = datetime.datetime(
        st_date.year, st_date.month, st_date.day, 6, 0, 0)  # 6 am
    st_date += datetime.timedelta(days=1)
    stop = datetime.datetime(
        st_date.year, st_date.month, st_date.day, 5, 59, 59)  # 6 am
    start_ts = start.timestamp() * 1000
    stop_ts = stop.timestamp() * 1000
    valid_files = [f if start_ts <= getctime(
        f) <= stop_ts else '' for f in flist]
    while '' in valid_files:
        valid_files.remove('')
    if len(valid_files) > 0:
        data_found = True
        main_flist[_st_date] = valid_files
        all_files += valid_files
        if first:
            print(_st_date, end='')
            first = False
        else:
            print(',', _st_date, end='')
        sys.stdout.flush()
if not data_found:
    print('None')
print('\n')

# %% Test image
idx = np.random.randint(0, high=len(flist))
img = np.asarray(pf.open(flist[idx])[1].data, dtype=float)

# %%
# encoding = {'imgs': {'dtype': float, 'zlib': True},
#             'exposure': {'dtype': float, 'zlib': True}}
# imgdata, expdata, tdata = get_imgs_from_files(all_files, wls[0])
# ds = xr.Dataset(
#             data_vars=dict(
#                 imgs=(['tstamp', 'height', 'wl'], imgdata),
#                 exposure=(['tstamp'], expdata)
#             ),
#             coords=dict(tstamp=tdata),
#             attrs=dict(wl=wls[0])
#         )
# fname = 'hitmis_night_%04d.nc'%(wls[0] * 10)
# print('Saving %s...\t' % (fname), end='')
# sys.stdout.flush()
# ds.to_netcdf(destdir + '/' + fname, encoding=encoding)
# print('Done.')
# sys.exit(0)

# %% Save NC files
encoding = {'imgs': {'dtype': float, 'zlib': True},
            'exposure': {'dtype': float, 'zlib': True}}
for key in main_flist.keys():
    filelist = main_flist[key]
    print('[%04d-%02d-%02d]' % (key.year, key.month, key.day))
    for w in wls:
        fname = f"{fprefix}_{key:%Y%m%d}_{w*10:04.0f}.nc"
        if fname in os.listdir(destdir) and not overwrite:
            print('File %s exists' % (fname))
            continue
        tstart = datetime.datetime.now().timestamp()
        imgdata, expdata, tdata = get_imgs_from_files(
            filelist, w, align_file, flatdir, do_flatfield=do_flatfield)
        tdelta = datetime.datetime.now().timestamp() - tstart
        try:
            print(' ' * os.get_terminal_size()[0], end='')
        except OSError:
            print(' ' * 80, end='')
        print('[%.1f] Conversion time: %s' % (w, tdelta_to_hms(tdelta)))
        wl_ax = np.arange(imgdata[0].shape[-1]) * \
            resolutions[w][0] + resolutions[w][1]
        imgdata = np.asarray(imgdata)
        expdata = np.asarray(expdata)
        imgdata /= expdata[:, None, None]
        ds = xr.Dataset(
            data_vars={
                'imgs': (('tstamp', 'height', 'wl'), imgdata[:, :, ::-1],
                         {'units': 'counts/nm/s'}),
                'exposure': (('tstamp',), expdata, {'units': 's'}),
            },
            coords={
                'tstamp': ('tstamp', tdata, {'units': 's', 'description': 'Unix timestamp'}),
                'height': ('height', (25 - np.arange(imgdata.shape[1]) * 0.12285012285012442),
                           {'units': 'deg'}),
                'wl': ('wl', wl_ax[::-1], {'units': 'nm'}),
            },
            attrs=dict(wl=w)
        )
        if do_flatfield:
            flat = flat_field_img(w, flatdir)
            ds['imgs'].values /= flat[None, :, :]
        # # do resamp
        # start = datetime.datetime.fromtimestamp(tdata[0])
        # end = start + datetime.timedelta(0, 240)
        # del imgdata
        # del expdata
        # del tdata
        # ts = []
        # imgs = []
        # stdvals = []
        # count = 0
        # str_len = 0
        # while end.timestamp() < ds['tstamp'][-1]:
        #     # resample
        #     slc = slice(start.timestamp(), end.timestamp())
        #     exps = ds.loc[dict(tstamp=slc)]['exposure']
        #     if exps is not None and len(exps) != 0:
        #         data = ds.loc[dict(tstamp=slc)]['imgs']
        #         data /= exps
        #         stdev = np.std(data, axis=0)
        #         data = np.average(data, axis=0)
        #         # save
        #         start += datetime.timedelta(0, 120)
        #         ts.append(start.timestamp())
        #         imgs.append(data)
        #         stdvals.append(stdev)
        #         count += 1
        #         p_str = '%s %d' % (str(end), count)
        #         if str_len:
        #             print(' '*str_len, end='\r')
        #         print('%s' % (p_str), end='\r')
        #         str_len = len(p_str)
        #     else:
        #         p_str = '%s: No files' % (str(end))
        #         if str_len:
        #             print(' '*str_len, end='\r')
        #         print('%s' % (p_str), end='\r')
        #         str_len = len(p_str)
        #         start += datetime.timedelta(0, 120)
        #     # update
        #     start += datetime.timedelta(0, 120)
        #     end += datetime.timedelta(0, 240)
        # del ds
        # imgs = np.asarray(imgs)
        # stdvals = np.asarray(stdvals)
        # print('Final:', len(imgs), len(ts))
        # encoding = {'imgs': {'dtype': float, 'zlib': True}}
        # nds = xr.Dataset(
        #     data_vars=dict(
        #         imgs=(['tstamp', 'height', 'wl'], imgs),
        #         stds=(['tstamp', 'height', 'wl'], stdvals),
        #     ),
        #     coords=dict(tstamp=ts, wl=wl_ax)
        # )
        print('Saving %s...\t' % (fname), end='')
        sys.stdout.flush()
        ds.to_netcdf(destdir + '/' + fname, encoding=encoding)
        print('Done.')

# %%
