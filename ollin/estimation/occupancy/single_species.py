from ..stanmodels import StanModel
from .base import OccupancyEstimate

CODE = u'''
data {
    int<lower=0> steps;
    int<lower=0> cams;
    int<lower=0, upper=1> detections[cams, steps];

    // Parameters for Ocuppancy Beta Prior
    real<lower=0> alpha_oc;
    real<lower=0> beta_oc;

    // Parameters for detectionsectability Beta Prior
    real<lower=0> alpha_det;
    real<lower=0> beta_det;
}
transformed data {
    int<lower=0> counts[cams]; // detectionsection count per site
    for(m in 1:cams) {
        counts[m] = sum(detections[m]);
    }
}
parameters {
    real<lower=0, upper=1> occupancy;
    real<lower=0, upper=1> detectability;
}
model {
    occupancy ~ beta(alpha_oc, beta_oc);
    detectability ~ beta(alpha_det, beta_det);

    for (j in 1:cams) {
        if (counts[j] == 0)
            target += log_sum_exp(
                binomial_lpmf(counts[j] | steps, detectability) + log(occupancy),
                log(1 - occupancy));
        else
            target += log(occupancy) + binomial_lpmf(counts[j] | steps, detectability);
    }
}
'''


class Model(StanModel):
    """McKenzie Model for Single Species - Single Season occupancy estimation.

    # TODO

    """

    name = 'McKenzie Single Species - Single Season'
    stancode = CODE

    def prepare_data(self, detection, priors):
        steps, cams = detection.detections.shape
        data = {
            'steps': steps,
            'cams': cams,
            'detections': detection.detections.T.astype(int).tolist(),
            'alpha_det': priors.get('alpha_det', 1),
            'beta_det': priors.get('beta_det', 1),
            'alpha_oc': priors.get('alpha_oc', 1),
            'beta_oc': priors.get('beta_oc', 1)}
        return data

    def estimate(self, detection, method='MAP', priors=None):
        stan_result = super(Model, self).estimate(detection, method, priors)
        occupancy = stan_result['occupancy']
        detectability = stan_result['detectability']
        est = OccupancyEstimate(
            occupancy, self, detection, detectability=detectability)
        return est
