from __future__ import division
import csv, warnings, copy, os, operator

import numpy as np
ma = np.ma
from matplotlib import verbose

import matplotlib.nxutils as nxutils
import matplotlib.cbook as cbook
from matplotlib import docstring


def griddata(x,y,z,xi,yi,interp):
    """
    ``zi = griddata(x,y,z,xi,yi)`` fits a surface of the form *z* =
    *f*(*x*, *y*) to the data in the (usually) nonuniformly spaced
    vectors (*x*, *y*, *z*).  :func:`griddata` interpolates this
    surface at the points specified by (*xi*, *yi*) to produce
    *zi*. *xi* and *yi* must describe a regular grid, can be either 1D
    or 2D, but must be monotonically increasing.

    A masked array is returned if any grid points are outside convex
    hull defined by input data (no extrapolation is done).

    If interp keyword is set to '`nn`' (default),
    uses natural neighbor interpolation based on Delaunay
    triangulation.  By default, this algorithm is provided by the
    :mod:`matplotlib.delaunay` package, written by Robert Kern.  The
    triangulation algorithm in this package is known to fail on some
    nearly pathological cases. For this reason, a separate toolkit
    (:mod:`mpl_tookits.natgrid`) has been created that provides a more
    robust algorithm fof triangulation and interpolation.  This
    toolkit is based on the NCAR natgrid library, which contains code
    that is not redistributable under a BSD-compatible license.  When
    installed, this function will use the :mod:`mpl_toolkits.natgrid`
    algorithm, otherwise it will use the built-in
    :mod:`matplotlib.delaunay` package.

    If the interp keyword is set to '`linear`', then linear interpolation
    is used instead of natural neighbor. In this case, the output grid
    is assumed to be regular with a constant grid spacing in both the x and
    y directions. For regular grids with nonconstant grid spacing, you
    must use natural neighbor interpolation.  Linear interpolation is only valid if
    :mod:`matplotlib.delaunay` package is used - :mod:`mpl_tookits.natgrid`
    only provides natural neighbor interpolation.

    The natgrid matplotlib toolkit can be downloaded from
    http://sourceforge.net/project/showfiles.php?group_id=80706&package_id=142792
    """
    try:
        from mpl_toolkits.natgrid import _natgrid, __version__
        _use_natgrid = True
    except ImportError:
        print 'IMPORT ERROR : PLEASE INSTALL NATGRID'
        import matplotlib.delaunay as delaunay
        from matplotlib.delaunay import  __version__
        _use_natgrid = False
    if not griddata._reported:
        if _use_natgrid:
            verbose.report('using natgrid version %s' % __version__)
        else:
            verbose.report('using delaunay version %s' % __version__)
        griddata._reported = True
 
    if xi.ndim != yi.ndim :
        raise TypeError("inputs xi and yi must have same number of dimensions (1 or 2)")
    if xi.ndim != 1 and xi.ndim != 2:
        raise TypeError("inputs xi and yi must be 1D or 2D.")
    if not len(x)==len(y)==len(z):
        raise TypeError("inputs x,y,z must all be 1D arrays of the same length")
    # remove masked points.
    if hasattr(z,'mask'):
        # make sure mask is not a scalar boolean array.
        if z.mask.ndim:
            x = x.compress(z.mask == False)
            y = y.compress(z.mask == False)
            z = z.compressed()
    if _use_natgrid: # use natgrid toolkit if available.
        if interp != 'nn':
            raise ValueError("only natural neighor interpolation"
            " allowed when using natgrid toolkit in griddata.")
        if xi.ndim == 2:
            xi = xi[0,:]
            yi = yi[:,0]
        # override default natgrid internal parameters.
        _natgrid.seti('ext',0)
        _natgrid.setr('nul',np.nan)
        # cast input arrays to doubles (this makes a copy)
        x = x.astype(np.float)
        y = y.astype(np.float)
        z = z.astype(np.float)
        xo = xi.astype(np.float)
        yo = yi.astype(np.float)
        if min(xo[1:]-xo[0:-1]) < 0 or min(yo[1:]-yo[0:-1]) < 0:
            raise ValueError, 'output grid defined by xi,yi must be monotone increasing'
        # allocate array for output (buffer will be overwritten by nagridd)
        zo = np.empty((yo.shape[0],xo.shape[0]), np.float)
        _natgrid.natgridd(x,y,z,xo,yo,zo)
    else: # use Robert Kern's delaunay package from scikits (default)
        if xi.ndim != yi.ndim:
            raise TypeError("inputs xi and yi must have same number of dimensions (1 or 2)")
        if xi.ndim != 1 and xi.ndim != 2:
            raise TypeError("inputs xi and yi must be 1D or 2D.")
        if xi.ndim == 1:
            xi,yi = np.meshgrid(xi,yi)
        # triangulate data
        tri = delaunay.Triangulation(x,y)
        # interpolate data
        if interp == 'nn':
            interp = tri.nn_interpolator(z)
            zo = interp(xi,yi)
        elif interp == 'linear':
            # make sure grid has constant dx, dy
            dx = xi[0,1:]-xi[0,0:-1]
            dy = yi[1:,0]-yi[0:-1,0]
            epsx = np.finfo(xi.dtype).resolution
            epsy = np.finfo(yi.dtype).resolution
            if dx.max()-dx.min() > epsx or dy.max()-dy.min() > epsy:
                raise ValueError("output grid must have constant spacing"
                                 " when using interp='linear'")
            interp = tri.linear_interpolator(z)
            zo = interp[yi.min():yi.max():complex(0,yi.shape[0]),
                        xi.min():xi.max():complex(0,xi.shape[1])]
        else:
            raise ValueError("interp keyword must be one of"
            " 'linear' (for linear interpolation) or 'nn'"
            " (for natural neighbor interpolation). Default is 'nn'.")
    # mask points on grid outside convex hull of input data.
    if np.any(np.isnan(zo)):
        zo = np.ma.masked_where(np.isnan(zo),zo)
    return zo
griddata._reported = False
