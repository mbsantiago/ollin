#**********************************************************************************************
#  fit species niche models based on one-class classification and outlier detection methods
#  

import numpy as np

import spatial_tools as st

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import MinCovDet




def fit_ellipsoid(train_table,type_of_cov="robust"):

	if (type_of_cov=="robust"):
		# fit a Minimum Covariance Determinant (MCD) robust estimator to data
		cov = MinCovDet().fit(train_table)

	if (type_of_cov=="empirical"):
		# estimators learnt from the full data set with true parameters
		cov = EmpiricalCovariance().fit(X)

	return(cov)


def predict_ellipsoid(predict_table,estimated_cov):
	
	distances = estimated_cov.mahalanobis(predict_table)

	return(distances)


