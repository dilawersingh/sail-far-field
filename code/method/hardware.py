import time
import numpy as np
from PIL import Image
from holoeye import slmdisplaysdk
from pypylon import pylon


def open_slm(required_version=5):
    """
    Open the HOLOEYE SLM and return the SLM instance.
    """
    slm = slmdisplaysdk.SLMInstance()

    if not slm.requiresVersion(required_version):
        raise RuntimeError(f"Required SDK version {required_version} not available.")

    error = slm.open()
    if error != slmdisplaysdk.ErrorCode.NoError:
        raise RuntimeError(slm.errorString(error))

    return slm


def open_camera():
    """
    Open the first available Basler camera using pypylon.
    """
    tlf = pylon.TlFactory.GetInstance()
    cam = pylon.InstantCamera(tlf.CreateFirstDevice())
    cam.Open()
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cam


def close_hardware(cam=None, slm=None):
    """
    Safely close camera and SLM.
    """
    if cam is not None:
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass

        try:
            cam.Close()
        except Exception:
            pass

    if slm is not None:
        try:
            slm.close()
        except Exception:
            pass


def tile_to_slm_centered(holo: np.ndarray, slm_shape=(1080, 1920)) -> np.ndarray:
    """
    Periodically tile a smaller 2D hologram onto the SLM such that one full copy
    of the hologram is centered on the SLM.
    """
    if holo.ndim != 2:
        raise ValueError(f"holo must be 2D, got shape {holo.shape}")

    h, w = holo.shape
    H, W = slm_shape

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid hologram shape: {holo.shape}")
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid SLM shape: {slm_shape}")

    yy, xx = np.indices((H, W))

    slm_cy = H // 2
    slm_cx = W // 2
    holo_cy = h // 2
    holo_cx = w // 2

    src_y = (yy - slm_cy + holo_cy) % h
    src_x = (xx - slm_cx + holo_cx) % w

    tiled = holo[src_y, src_x]
    return np.ascontiguousarray(tiled)


def flush_camera(cam, n_flush=2, timeout_ms=5000):
    """
    Discard a few buffered frames so the next grab is fresh.
    """
    for _ in range(n_flush):
        result = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            pass
        finally:
            result.Release()


def grab_frame(cam, timeout_ms=5000):
    """
    Grab one raw camera frame as a NumPy array.
    """
    result = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    try:
        if not result.GrabSucceeded():
            raise RuntimeError("Camera grab failed.")
        return np.array(result.Array, copy=True)
    finally:
        result.Release()


def crop_np(img: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """
    Crop a 2D image using top-left origin (y0, x0) and size (h, w).
    """
    H, W = img.shape[:2]
    y1 = y0 + h
    x1 = x0 + w

    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        raise ValueError(
            f"Requested crop (y0={y0}, x0={x0}, h={h}, w={w}) "
            f"is outside image shape {img.shape}"
        )

    return img[y0:y1, x0:x1]


def center_crop_np(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Center crop a 2D image to (out_h, out_w).
    """
    h, w = img.shape[:2]

    if out_h > h or out_w > w:
        raise ValueError(f"Requested crop {(out_h, out_w)} exceeds image shape {img.shape}")

    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return img[y0:y0 + out_h, x0:x0 + out_w]


def resize_np(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resize a 2D image to (out_h, out_w).
    """
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(pil_img)


def normalize_sum_np(img: np.ndarray, target_sum: float = None, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a 2D image so that its sum equals target_sum.
    If target_sum is None, use number of pixels, giving mean ~ 1.
    """
    img = img.astype(np.float32, copy=False)
    s = float(img.sum())

    if target_sum is None:
        target_sum = float(img.shape[0] * img.shape[1])

    return img * (target_sum / max(s, eps))


def grab_preprocessed_np(
    cam,
    roi=None,
    out_h=None,
    out_w=None,
    resize_to_out=True,
    normalize_sum=True,
    timeout_ms=5000,
):
    """
    Workflow:
        raw frame -> ROI crop -> optional resize to (out_h, out_w) -> normalize

    Parameters
    ----------
    cam : pypylon camera
    roi : dict or None
        Example:
            {"y0": 0, "x0": 360, "h": 1200, "w": 1200}
    out_h, out_w : int or None
        Desired final output size.
    resize_to_out : bool
        If True and out_h/out_w are provided, resize ROI to output size.
    normalize_sum : bool
        If True, normalize output intensity sum.
    timeout_ms : int
        Camera timeout in milliseconds.
    """
    frame = grab_frame(cam, timeout_ms=timeout_ms)

    if frame.ndim != 2:
        raise ValueError(f"Expected mono camera frame, got shape {frame.shape}")

    if roi is not None:
        frame = crop_np(
            frame,
            y0=roi["y0"],
            x0=roi["x0"],
            h=roi["h"],
            w=roi["w"],
        )

    if out_h is not None and out_w is not None and resize_to_out:
        frame = resize_np(frame, out_h=out_h, out_w=out_w)

    frame = frame.astype(np.float32)

    if normalize_sum:
        if out_h is not None and out_w is not None and resize_to_out:
            target_sum = out_h * out_w
        else:
            target_sum = frame.shape[0] * frame.shape[1]

        frame = normalize_sum_np(frame, target_sum=target_sum)

    return frame


def show_on_slm_and_grab(
    slm,
    cam,
    holo_small_uint8: np.ndarray,
    slm_shape=(1080, 1920),
    settle_s=2.0,
    n_flush=2,
    roi=None,
    cam_out_h=None,
    cam_out_w=None,
    resize_to_out=True,
    normalize_sum=True,
    timeout_ms=5000,
):
    """
    Center-tile a smaller hologram to the full SLM, display it, wait for settling,
    flush buffered camera frames, then grab one synchronized processed frame.

    Returns
    -------
    holo_slm : np.ndarray
        Full SLM pattern actually displayed.
    frame : np.ndarray
        Processed camera frame.
    """
    holo_slm = tile_to_slm_centered(holo_small_uint8, slm_shape=slm_shape)
    holo_slm = np.ascontiguousarray(holo_slm.astype(np.uint8))

    error = slm.showData(holo_slm)
    if error != slmdisplaysdk.ErrorCode.NoError:
        raise RuntimeError(f"SLM error: {slm.errorString(error)}")

    time.sleep(settle_s)
    flush_camera(cam, n_flush=n_flush, timeout_ms=timeout_ms)

    frame = grab_preprocessed_np(
        cam,
        roi=roi,
        out_h=cam_out_h,
        out_w=cam_out_w,
        resize_to_out=resize_to_out,
        normalize_sum=normalize_sum,
        timeout_ms=timeout_ms,
    )

    return holo_slm, frame