# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""quick test script for neurova.nvc module."""

import numpy as np
from neurova import nvc

print('testing neurova.nvc module...')

# test image creation
img = np.zeros((100, 100, 3), dtype=np.uint8)

# test drawing
nvc.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
nvc.circle(img, (50, 50), 30, (255, 0, 0), -1)
nvc.line(img, (0, 0), (100, 100), (0, 0, 255), 1)
print('  drawing functions: ok')

# test color conversion
gray = nvc.cvtColor(img, nvc.COLOR_BGR2GRAY)
print('  color conversion: ok')

# test thresholding
_, thresh = nvc.threshold(gray, 127, 255, nvc.THRESH_BINARY)
print('  thresholding: ok')

# test contours
contours, hierarchy = nvc.findContours(thresh, nvc.RETR_EXTERNAL, nvc.CHAIN_APPROX_SIMPLE)
print(f'  contours found: {len(contours)}')

# test filters
blurred = nvc.GaussianBlur(img, (5, 5), sigma=1.0)
print('  gaussian blur: ok')

# test morphology
kernel = nvc.getStructuringElement(nvc.MORPH_RECT, (3, 3))
dilated = nvc.dilate(thresh, kernel)
print('  morphology: ok')

# test new functions
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = nvc.hconcat([a, b])
print('  hconcat: ok')

det = nvc.determinant(a.astype(np.float64))
print(f'  determinant: {det}')

# test trackers exist
print(f'  TrackerMIL: {nvc.TrackerMIL}')
print(f'  TrackerKCF: {nvc.TrackerKCF}')
print(f'  TrackerCSRT: {nvc.TrackerCSRT}')

# nvc module is primary API
print(f'  nvc module available: {nvc is not None}')

print('')
print('all tests passed!')

# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.
