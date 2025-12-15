

import numpy as np


tennis_ball = """
[267 231]
[268 231]
[269 231]
[266 232]
[267 232]
[268 232]
[269 232]
[266 233]
[267 233]
[268 233]
[269 233]
[266 234]
[267 234]
[268 234]
"""

mask_tennis_ball = np.fromstring(tennis_ball.replace('[','').replace(']',''),
                                               sep=' ').reshape(-1, 2)
