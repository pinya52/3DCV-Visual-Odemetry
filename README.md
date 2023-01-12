# 3DCV-Visual-Odemetry

## Reproduce my work
### Camera calibration
```shell
# To obtain camera intrinsic matrix and distortion coefficients
bash calibrate.sh
```

### Reconstruct the camera trajectory
```shell
# The default method to detect feature points is ORB
python3 vo.py frames

# Using SIFT as a different method
python3 vo.py frames --method sift
```
