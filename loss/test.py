#([4046,11198])
#([4046,36916])

import torch.nn as nn
import torch
def _pad(scores, teacher_scores):

    if scores.size(1) > teacher_scores.size(1):
            # teacher_scores must be padded to the same size to scores
        pad_size = int(scores.size(1)-teacher_scores.size(1))
        if pad_size % 2 == 0:
            pad_range = (pad_size/2, pad_size/2)
        else:
            pad_range = (pad_size//2, (pad_size//2)+1)
        teacher_scores = nn.functional.pad(teacher_scores, pad_range)
        return scores, teacher_scores

    elif teacher_scores.size(1) > scores.size(1):
        pad_size = int(teacher_scores.size(1) - scores.size(1))
        if pad_size % 2 == 0:
            pad_range = (int(pad_size/2), int(pad_size/2))
        else:
            pad_range = (pad_size//2, (pad_size//2)+1)
        scores = nn.functional.pad(scores, pad_range)
        return scores, teacher_scores

    else:
        return scores, teacher_scores

if __name__ == "__main__":
    #([4046,11198])
    #([4046,36916])
    a = torch.full([4046,11198],0)
    b = torch.full([4046,36916],0)

    _a,_b =_pad(a,b)

    pass