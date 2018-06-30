data {
    int<lower=0> Days; // Number of days
    int<lower=0> Cams; // Number of cameras
    int<lower=0, upper=1> Det[Cams, Days]; // Detection matrix

    // Parameters for Ocuppancy Beta Prior
    real<lower=0> alpha_oc;
    real<lower=0> beta_oc;

    // Parameters for Detectability Beta Prior
    real<lower=0> alpha_det;
    real<lower=0> beta_det;
}
transformed data {
    int<lower=0> Counts[Cams]; // Detection count per site
    for(m in 1:Cams) {
        Counts[m] = sum(Det[m]);
    }
}
parameters {
    real<lower=0, upper=1> occupancy;
    real<lower=0, upper=1> detectability;
}
model {
    occupancy ~ beta(alpha_oc, beta_oc);
    detectability ~ beta(alpha_det, beta_det);

    for (j in 1:Cams) {
        if (Counts[j] == 0)
            target += log_sum_exp(
                binomial_lpmf(Counts[j] | Days, detectability) + log(occupancy),
                log(1 - occupancy));
        else
            target += log(occupancy) + binomial_lpmf(Counts[j] | Days, detectability);
    }
}
