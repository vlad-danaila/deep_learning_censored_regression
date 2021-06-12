import torch as t
import math

mae = t.nn.L1Loss()

def truncation_mae_penalty(y_pred, lower_truncation_limit = None, upper_truncation_limit = None):
  lower_truncation_penalty, upper_truncation_penalty = 0, 0
  if lower_truncation_limit:
    below_lower_truncation_limit = t.clamp(y_pred, min = -math.inf, max = lower_truncation_limit)
    lower_truncation_penalty = mae(below_lower_truncation_limit, t.full_like(below_lower_truncation_limit, lower_truncation_limit))
  if upper_truncation_limit:
    above_upper_truncation_limit = t.clamp(y_pred, min = upper_truncation_limit, max = math.inf)
    upper_truncation_penalty = mae(above_upper_truncation_limit, t.full_like(above_upper_truncation_limit, upper_truncation_limit))
  return lower_truncation_penalty + upper_truncation_penalty

def censored_mae(y_pred, y, lower_censoring_bound = -math.inf, upper_censoring_bound = math.inf, device = 'cpu', clamp_all = False):
  if clamp_all:
    y_pred = t.clamp(y_pred, min = lower_censoring_bound, max = upper_censoring_bound)
  else:
    y_pred = t.where((y > lower_censoring_bound) + (y_pred > lower_censoring_bound), y_pred, t.tensor(lower_censoring_bound, device = device))
    y_pred = t.where((y < upper_censoring_bound) + (y_pred < upper_censoring_bound), y_pred, t.tensor(upper_censoring_bound, device = device))
  return mae(y_pred, y)

def censored_mae_with_truncation_penalty(
        y_pred,
        y,
        lower_censoring_bound = -math.inf,
        upper_censoring_bound = math.inf,
        lower_truncation_limit = None,
        upper_truncation_limit = None,
        device = 'cpu',
        clamp_all = False
  ):
  return censored_mae(y_pred, y, lower_censoring_bound = lower_censoring_bound, upper_censoring_bound = upper_censoring_bound, device = device, clamp_all = clamp_all) \
         + truncation_mae_penalty(y_pred, lower_truncation_limit = lower_truncation_limit, upper_truncation_limit = upper_truncation_limit)



