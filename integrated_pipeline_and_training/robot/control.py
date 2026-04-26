def compute_error(needle, target):
    xn, yn = needle
    xt, yt = target
    return xt - xn, yt - yn


def clip(value, max_step):
    return max(-max_step, min(max_step, value))