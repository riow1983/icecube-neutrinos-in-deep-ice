import torch


# losses.py
def angular_dist_score(y_pred, y_true):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two itorchut vectors

    # https://www.kaggle.com/code/sohier/mean-angular-error

    Parameters:
    -----------

    y_pred : float (torch.Tensor)
        Prediction array of [N, 2], where the second dim is azimuth & zenith
    y_true : float (torch.Tensor)
        Ground truth array of [N, 2], where the second dim is azimuth & zenith

    Returns:
    --------

    dist : float (torch.Tensor)
        mean over the angular distance(s) in radian
    """

    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = torch.clamp(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))
