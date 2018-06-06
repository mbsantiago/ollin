#**********************************************************************************************
#  concentrates raster and shapefile input, output, manipulation and interaction capabilities
#            

import sys

import os

import struct 

import numpy as np

import pandas as pd

from osgeo import gdal
from osgeo import ogr

from osgeo.gdalconst import GA_ReadOnly 
from osgeo.gdalconst import GDT_Float32
from osgeo.gdalconst import GDT_Int16

def readtif(imagepath):
    gdal.AllRegister()
    inDataset = gdal.Open(imagepath,GA_ReadOnly)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    bands = inDataset.RasterCount
    return(inDataset,rows,cols,bands)

def createtif(driver,rows,cols,bands,outpath,data_type=32):
    if data_type==32:
        outDataset = driver.Create(outpath,cols,rows,bands,GDT_Float32)
    elif data_type==16:
        outDataset = driver.Create(outpath,cols,rows,bands,GDT_Int16)
    return(outDataset)

def writetif(outDataset,data,projection,geotransform,order='r'):
    # order controls if the columns or the rows should be considered the observations
    cols = outDataset.RasterXSize
    rows = outDataset.RasterYSize 

    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + 0*gt[1] # 0 for now becasue we want no shift
        gt[3] = gt[3] + 0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)
    
    if data.ndim==1:
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(np.resize(data,(rows,cols)),0,0)
        outBand.FlushCache()
    else:
        if order=='r':
            n=np.shape(data)[0]
            for k in range(n):
                outBand = outDataset.GetRasterBand(k+1)
                outBand.WriteArray(np.resize(data[k,:],(rows,cols)),0,0)
                outBand.FlushCache()
        elif order=='c':
            n=np.shape(data)[1]
            for k in range(n):
                outBand = outDataset.GetRasterBand(k+1)
                outBand.WriteArray(np.reshape(data[:,k],(rows,cols)))
                outBand.FlushCache()

    # close the dataset properly
    outDataset = None

def list_files(directory, extension="tif"):

    files = []
    for f in os.listdir(directory):
        if f.endswith('.' + extension):
            files.append(f)
    return(files)

def shapecoordinates(shape,output="dataframe"):
    
    '''
    
    Extract coordinates of each object of a shapefile
    and return as pandas dataframe or numpy array
    
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds=driver.Open(shape,0)
    layer = ds.GetLayer()

    # number of objects in shapefile
    count = layer.GetFeatureCount()

    # initialize numpy array to fill with objectid, xcoord and ycoord
    coordinates = np.zeros((3,count))

    for i in xrange(count):

        feature = layer.GetFeature(i)
        geom = feature.GetGeometryRef()
        mx,my = geom.GetX(), geom.GetY()
        
        coordinates[0,i] = i
        coordinates[1,i] = mx
        coordinates[2,i] = my

    if output=="dataframe":
        index = range(count)
        columns = ["id","x","y"]
        coordinates=pd.DataFrame.from_records(data=coordinates[[0,1,2],:].T,index=index,columns=columns)
    
    return coordinates

    # flush memory
    feature.Destroy()
    ds.Destroy()

def extract(xcoord,ycoord,image,nodatavalue=None,data_type=32):

    '''
    extract raster-value at a given location    
    
    '''
    if data_type == 32:
        data_type = gdal.GDT_Float32
        format ="f"
    else:
        data_type = gdal.GDT_Int16
        format ="h"


    # read image
    dataset,rows,cols,bands = readtif(image)

    # image metadata
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    driver = dataset.GetDriver()

    xcoord = int((xcoord - geotransform[0]) / geotransform[1])
    ycoord = int((ycoord - geotransform[3]) / geotransform[5])

    # pixel value
    band=dataset.GetRasterBand(1)
    binaryvalue=band.ReadRaster(xcoord,ycoord,1,1,buf_type=data_type)
    value = struct.unpack(format, binaryvalue)
    value = value[0]

    if value == nodatavalue:
        value = np.nan

    return value

def traintable_from_rasters(path_to_shapefile, path_to_covariate_rasters,remove_nans=True):
    
    '''
    
    Given a point shapefile and a collection of rasters produces an extraction of
    the rasters using the points (as of now only thought of for one-class classification)
    
    '''

    # coordinates of each object in shapefile
    shpcoordinates = shapecoordinates(path_to_shapefile)

    # list of rasters in path
    raster_files = list_files(path_to_covariate_rasters)

    # dimensions of output
    n_out_rows = np.shape(shpcoordinates)[0]
    n_out_cols = len(raster_files)

    # initialize output numpy array
    output = np.zeros((n_out_rows,n_out_cols),dtype=np.float64)

    for i in xrange(n_out_cols):

        file = raster_files[i]

        imagepath=path_to_covariate_rasters+file

        for j in xrange(n_out_rows):

            value = extract(shpcoordinates.iloc[j,1],shpcoordinates.iloc[j,2],\
                            image=imagepath,\
                            data_type=32)

            output[j,i] = value

    if remove_nans:

        output = np.nan_to_num(output,copy=False)
    
    return output

def predictiontable_from_rasters(path_to_covariate_rasters,remove_nans=True):
    
    '''
    
    Given collection of rasters creates a numpy array of flattened covariates (one column per raster)

    warning: all extents and projections must match
    
    '''

    # list of rasters in path
    raster_files = list_files(path_to_covariate_rasters)

    # read a raster to serve as base

    dataset,rows,cols,bands = readtif(path_to_covariate_rasters+raster_files[0])

    # initialize output numpy array
    output = np.zeros((rows*cols,len(raster_files)),dtype=np.float64)

    for i in xrange(len(raster_files)):

        dataset,rows,cols,bands = readtif(path_to_covariate_rasters+raster_files[i])

        # make numpy array and flatten
        band = dataset.GetRasterBand(1)
        band = band.ReadAsArray(0, 0, cols, rows).astype(np.int64)
        band = np.ravel(band)

        output[:,i]=band

    if remove_nans:

        output = np.nan_to_num(output,copy=False)
    
    return output