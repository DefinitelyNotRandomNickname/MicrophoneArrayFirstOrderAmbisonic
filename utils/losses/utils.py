def reduce_loss(loss, reduction="mean"):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
    

def weighted_reduce_loss(loss, weights=None, reduction="mean", eps=1e-8):
    if weights is None:
        return reduce_loss(loss, reduction)

    weighted_loss = loss * weights
    if reduction == "mean":
        return weighted_loss.sum() / weights.sum().clamp_min(eps)
    elif reduction == "sum":
        return weighted_loss.sum()
    return weighted_loss
