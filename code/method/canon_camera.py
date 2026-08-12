from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------
# EDSDK basic typedefs
# -----------------------------------------------------------------------------
EdsError = ctypes.c_uint32
EdsBool = ctypes.c_int32
EdsUInt32 = ctypes.c_uint32
EdsInt32 = ctypes.c_int32
EdsInt64 = ctypes.c_int64
EdsChar = ctypes.c_char
EdsVoid = None
EdsBaseRef = ctypes.c_void_p
EdsCameraListRef = ctypes.c_void_p
EdsCameraRef = ctypes.c_void_p
EdsDirectoryItemRef = ctypes.c_void_p
EdsStreamRef = ctypes.c_void_p
EdsContext = ctypes.c_void_p
EdsDataType = ctypes.c_uint32  # out-param type for EdsGetPropertySize; value itself isn't needed here


# -----------------------------------------------------------------------------
# Common EDSDK constants.
# NOTE:
# These values are standard for recent EDSDK releases, but Canon occasionally
# adds newer symbols. If your local SDK headers differ, prefer your headers.
# -----------------------------------------------------------------------------
EDS_ERR_OK = 0x00000000
EDS_ERR_DEVICE_BUSY = 0x00000081
EDS_ERR_OBJECT_NOTREADY = 0x000000A102 if False else 0x000000A1  # kept for readability; actual uses vary by SDK

# Object events
kEdsObjectEvent_All = 0x00000200
kEdsObjectEvent_DirItemRequestTransfer = 0x00000208
kEdsObjectEvent_DirItemRequestTransferDT = 0x00000209

# Camera commands
kEdsCameraCommand_TakePicture = 0x00000000

# File create disposition / access / stream types
kEdsFileCreateDisposition_CreateAlways = 0
kEdsAccess_ReadWrite = 2
kEdsFileStreamType_Normal = 0

# Save destination and capacity-related properties
kEdsPropID_SaveTo = 0x0000000B
kEdsSaveTo_Host = 2

# Exposure properties (EDSDK API Programming Reference, section 5.2).
# All three are Read/Write EdsUInt32 on EdsCameraRef -- values are the
# hex-coded lookup-table codes from the manual, NOT the human values
# themselves (e.g. Tv=0x64 means 1/45s, not "0x64 seconds").
kEdsPropID_AEMode = 0x00000400      # shooting mode dial position -- READ ONLY on cameras
                                    # with a physical mode dial (1000D, T2i both qualify);
                                    # must be turned to M by hand for Av/Tv writes below
                                    # to actually take effect.
kEdsPropID_ISOSpeed = 0x00000402
kEdsPropID_Av = 0x00000405
kEdsPropID_Tv = 0x00000406

kEdsAEMode_Manual = 0x03  # value of kEdsPropID_AEMode when the dial is on M

# This is the classic structure used with EdsSetCapacity.
# The exact field names match Canon headers.
class EdsCapacity(ctypes.Structure):
    _fields_ = [
        ("numberOfFreeClusters", EdsInt32),
        ("bytesPerSector", EdsInt32),
        ("reset", EdsBool),
    ]


class EdsDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("szPortName", EdsChar * 256),
        ("szDeviceDescription", EdsChar * 256),
        ("deviceSubType", EdsUInt32),
        ("reserved", EdsUInt32),
    ]


class EdsDirectoryItemInfo(ctypes.Structure):
    _fields_ = [
        ("size", EdsUInt32),
        ("isFolder", EdsBool),
        ("groupID", EdsUInt32),
        ("option", EdsUInt32),
        ("szFileName", EdsChar * 256),
        ("format", EdsUInt32),
        ("dateTime", EdsUInt32),
    ]


# Callback types
ObjectEventHandler = ctypes.WINFUNCTYPE(EdsError, EdsUInt32, EdsBaseRef, EdsContext)


class CanonEDSDKError(RuntimeError):
    def __init__(self, message: str, code: int | None = None):
        self.code = code
        suffix = f" (0x{code:08X})" if code is not None else ""
        super().__init__(message + suffix)


@dataclass
class CaptureResult:
    path: Path
    elapsed_s: float


class CanonEDSDKCamera:
    """
    Thin ctypes wrapper around Canon EDSDK for remote still capture.

    Intended usage:
        camera = CanonEDSDKCamera(
            dll_path="path/to/EDSDK.dll",
            save_dir="path/to/captures",
        )
        camera.initialize()
        camera.open_session()
        path = camera.capture_image().path
        camera.close()

    Design notes:
    - Keeps a persistent SDK + camera session for the full training run.
    - Uses an object event callback to receive transfer requests.
    - Downloads the captured file to save_dir.
    - Polls EdsGetEvent() while waiting, which Canon recommends for
      console-style applications.
    """

    def __init__(
        self,
        dll_path: str | Path,
        save_dir: str | Path,
        *,
        auto_set_save_to_host: bool = False,
        auto_set_capacity: bool = False,
        verbose: bool = True,
    ) -> None:
        self.dll_path = Path(dll_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.auto_set_save_to_host = auto_set_save_to_host
        self.auto_set_capacity = auto_set_capacity
        self.verbose = verbose

        self._sdk: Optional[ctypes.WinDLL] = None
        self._camera_list: EdsCameraListRef = EdsCameraListRef()
        self._camera: EdsCameraRef = EdsCameraRef()
        self._session_open = False
        self._initialized = False

        self._capture_event = threading.Event()
        self._capture_error: Optional[BaseException] = None
        self._last_capture_path: Optional[Path] = None
        self._capture_counter = 0

        # Keep callback alive for lifetime of object.
        self._object_cb = ObjectEventHandler(self._handle_object_event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        if self._initialized:
            return

        if not self.dll_path.exists():
            raise FileNotFoundError(f"EDSDK DLL not found: {self.dll_path}")

        self._sdk = ctypes.WinDLL(str(self.dll_path))
        self._bind_functions()

        self._check(self._sdk.EdsInitializeSDK(), "EdsInitializeSDK failed")
        self._initialized = True

        self._camera_list = EdsCameraListRef()
        self._check(
            self._sdk.EdsGetCameraList(ctypes.byref(self._camera_list)),
            "EdsGetCameraList failed",
        )

        count = EdsUInt32()
        self._check(
            self._sdk.EdsGetChildCount(self._camera_list, ctypes.byref(count)),
            "EdsGetChildCount(camera_list) failed",
        )
        if count.value < 1:
            raise CanonEDSDKError("No Canon camera detected by EDSDK")

        self._camera = EdsCameraRef()
        self._check(
            self._sdk.EdsGetChildAtIndex(self._camera_list, 0, ctypes.byref(self._camera)),
            "EdsGetChildAtIndex(camera_list, 0) failed",
        )

        info = EdsDeviceInfo()
        self._check(
            self._sdk.EdsGetDeviceInfo(self._camera, ctypes.byref(info)),
            "EdsGetDeviceInfo failed",
        )
        if self.verbose:
            model = bytes(info.szDeviceDescription).split(b"\x00", 1)[0].decode(errors="ignore")
            port = bytes(info.szPortName).split(b"\x00", 1)[0].decode(errors="ignore")
            print(f"[CanonEDSDK] Connected camera: {model or '<unknown>'} on {port or '<unknown>'}")

    def open_session(self) -> None:
        self._require_initialized()
        if self._session_open:
            return

        self._check(self._sdk.EdsOpenSession(self._camera), "EdsOpenSession failed")
        self._session_open = True

        self._check(
            self._sdk.EdsSetObjectEventHandler(
                self._camera,
                kEdsObjectEvent_All,
                self._object_cb,
                None,
            ),
            "EdsSetObjectEventHandler failed",
        )

        if self.auto_set_save_to_host:
            self._set_save_to_host()
        if self.auto_set_capacity:
            self._set_unlimited_capacity()

        if self.verbose:
            print("[CanonEDSDK] Session opened")

    def capture_image(self, timeout_s: float = 10.0, poll_interval_s: float = 0.01) -> CaptureResult:
        self._require_session()

        self._capture_counter += 1
        self._capture_event.clear()
        self._capture_error = None
        self._last_capture_path = None

        t0 = time.perf_counter()
        self._send_take_picture()

        deadline = t0 + timeout_s
        while time.perf_counter() < deadline:
            # Canon explicitly documents calling this regularly in console apps.
            self._check(self._sdk.EdsGetEvent(), "EdsGetEvent failed during capture wait")

            if self._capture_error is not None:
                raise self._capture_error

            if self._capture_event.wait(timeout=poll_interval_s):
                if self._last_capture_path is None:
                    raise CanonEDSDKError("Capture signaled complete but no file path was recorded")
                return CaptureResult(path=self._last_capture_path, elapsed_s=time.perf_counter() - t0)

        raise TimeoutError(f"Timed out waiting {timeout_s:.2f}s for camera capture event")

    def close(self) -> None:
        if self._sdk is None:
            return

        try:
            if self._session_open and self._camera:
                try:
                    self._sdk.EdsCloseSession(self._camera)
                except Exception:
                    pass
                self._session_open = False
        finally:
            if self._camera:
                try:
                    self._sdk.EdsRelease(self._camera)
                except Exception:
                    pass
                self._camera = EdsCameraRef()

            if self._camera_list:
                try:
                    self._sdk.EdsRelease(self._camera_list)
                except Exception:
                    pass
                self._camera_list = EdsCameraListRef()

            if self._initialized:
                try:
                    self._sdk.EdsTerminateSDK()
                except Exception:
                    pass
                self._initialized = False

            if self.verbose:
                print("[CanonEDSDK] Closed session and terminated SDK")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bind_functions(self) -> None:
        assert self._sdk is not None
        dll = self._sdk

        dll.EdsInitializeSDK.restype = EdsError
        dll.EdsTerminateSDK.restype = EdsError

        dll.EdsGetCameraList.argtypes = [ctypes.POINTER(EdsCameraListRef)]
        dll.EdsGetCameraList.restype = EdsError

        dll.EdsGetChildCount.argtypes = [EdsBaseRef, ctypes.POINTER(EdsUInt32)]
        dll.EdsGetChildCount.restype = EdsError

        dll.EdsGetChildAtIndex.argtypes = [EdsBaseRef, EdsInt32, ctypes.POINTER(EdsBaseRef)]
        dll.EdsGetChildAtIndex.restype = EdsError

        dll.EdsGetDeviceInfo.argtypes = [EdsCameraRef, ctypes.POINTER(EdsDeviceInfo)]
        dll.EdsGetDeviceInfo.restype = EdsError

        dll.EdsOpenSession.argtypes = [EdsCameraRef]
        dll.EdsOpenSession.restype = EdsError

        dll.EdsCloseSession.argtypes = [EdsCameraRef]
        dll.EdsCloseSession.restype = EdsError

        dll.EdsSetObjectEventHandler.argtypes = [EdsCameraRef, EdsUInt32, ObjectEventHandler, EdsContext]
        dll.EdsSetObjectEventHandler.restype = EdsError

        dll.EdsSendCommand.argtypes = [EdsCameraRef, EdsUInt32, EdsUInt32]
        dll.EdsSendCommand.restype = EdsError

        dll.EdsGetEvent.argtypes = []
        dll.EdsGetEvent.restype = EdsError

        dll.EdsGetDirectoryItemInfo.argtypes = [EdsDirectoryItemRef, ctypes.POINTER(EdsDirectoryItemInfo)]
        dll.EdsGetDirectoryItemInfo.restype = EdsError

        dll.EdsCreateFileStream.argtypes = [
            ctypes.c_char_p,
            EdsUInt32,
            EdsUInt32,
            ctypes.POINTER(EdsStreamRef),
        ]
        dll.EdsCreateFileStream.restype = EdsError

        dll.EdsDownload.argtypes = [EdsDirectoryItemRef, EdsUInt32, EdsStreamRef]
        dll.EdsDownload.restype = EdsError

        dll.EdsDownloadComplete.argtypes = [EdsDirectoryItemRef]
        dll.EdsDownloadComplete.restype = EdsError

        dll.EdsDownloadCancel.argtypes = [EdsDirectoryItemRef]
        dll.EdsDownloadCancel.restype = EdsError

        dll.EdsRelease.argtypes = [EdsBaseRef]
        dll.EdsRelease.restype = EdsUInt32

        dll.EdsSetPropertyData.argtypes = [EdsBaseRef, EdsUInt32, EdsInt32, EdsUInt32, ctypes.c_void_p]
        dll.EdsSetPropertyData.restype = EdsError

        dll.EdsGetPropertySize.argtypes = [
            EdsBaseRef, EdsUInt32, EdsInt32, ctypes.POINTER(EdsDataType), ctypes.POINTER(EdsUInt32)
        ]
        dll.EdsGetPropertySize.restype = EdsError

        dll.EdsGetPropertyData.argtypes = [EdsBaseRef, EdsUInt32, EdsInt32, EdsUInt32, ctypes.c_void_p]
        dll.EdsGetPropertyData.restype = EdsError

        # Some cameras need this when SaveTo=Host.
        if hasattr(dll, 'EdsSetCapacity'):
            dll.EdsSetCapacity.argtypes = [EdsCameraRef, EdsCapacity]
            dll.EdsSetCapacity.restype = EdsError

    def _send_take_picture(self) -> None:
        assert self._sdk is not None

        last_err = None
        for _ in range(5):
            err = self._sdk.EdsSendCommand(self._camera, kEdsCameraCommand_TakePicture, 0)
            if err == EDS_ERR_OK:
                return
            last_err = err
            if err == EDS_ERR_DEVICE_BUSY:
                time.sleep(0.2)
                continue
            break

        raise CanonEDSDKError("EdsSendCommand(TakePicture) failed", last_err)

    def _handle_object_event(self, in_event: int, in_ref: int, in_context: int) -> int:
        # This callback executes on the SDK/session thread. Keep it short.
        try:
            if in_event in (kEdsObjectEvent_DirItemRequestTransfer, kEdsObjectEvent_DirItemRequestTransferDT):
                self._download_dir_item(EdsDirectoryItemRef(in_ref))
        except BaseException as exc:  # noqa: BLE001
            self._capture_error = exc
            self._capture_event.set()
        finally:
            # Canon says event-created objects should be released when no longer needed.
            if in_ref and self._sdk is not None:
                try:
                    self._sdk.EdsRelease(EdsBaseRef(in_ref))
                except Exception:
                    pass
        return EDS_ERR_OK

    def _download_dir_item(self, dir_item_ref: EdsDirectoryItemRef) -> None:
        assert self._sdk is not None

        info = EdsDirectoryItemInfo()
        self._check(
            self._sdk.EdsGetDirectoryItemInfo(dir_item_ref, ctypes.byref(info)),
            "EdsGetDirectoryItemInfo failed",
        )

        raw_name = bytes(info.szFileName).split(b"\x00", 1)[0].decode(errors="ignore") or "capture.jpg"
        suffix = Path(raw_name).suffix or ".jpg"
        safe_name = f"capture_{self._capture_counter:06d}{suffix}"
        out_path = self.save_dir / safe_name

        stream = EdsStreamRef()
        self._check(
            self._sdk.EdsCreateFileStream(
                str(out_path).encode("mbcs"),
                kEdsFileCreateDisposition_CreateAlways,
                kEdsAccess_ReadWrite,
                ctypes.byref(stream),
            ),
            f"EdsCreateFileStream failed for {out_path}",
        )

        try:
            self._check(
                self._sdk.EdsDownload(dir_item_ref, info.size, stream),
                "EdsDownload failed",
            )
            self._check(
                self._sdk.EdsDownloadComplete(dir_item_ref),
                "EdsDownloadComplete failed",
            )
        except Exception:
            try:
                self._sdk.EdsDownloadCancel(dir_item_ref)
            except Exception:
                pass
            raise
        finally:
            if stream:
                try:
                    self._sdk.EdsRelease(stream)
                except Exception:
                    pass

        self._last_capture_path = out_path
        self._capture_event.set()

        if self.verbose:
            print(f"[CanonEDSDK] Downloaded: {out_path}")

    def _set_save_to_host(self) -> None:
        assert self._sdk is not None
        value = EdsUInt32(kEdsSaveTo_Host)
        self._check(
            self._sdk.EdsSetPropertyData(
                self._camera,
                kEdsPropID_SaveTo,
                0,
                ctypes.sizeof(value),
                ctypes.byref(value),
            ),
            "EdsSetPropertyData(SaveTo=Host) failed",
        )
        if self.verbose:
            print("[CanonEDSDK] Save destination set to host")

    # ------------------------------------------------------------------
    # Exposure property get/set (ISO / Av / Tv)
    # ------------------------------------------------------------------
    def _get_property_uint32(self, prop_id: int) -> int:
        assert self._sdk is not None
        data_type = EdsDataType()
        data_size = EdsUInt32()
        self._check(
            self._sdk.EdsGetPropertySize(
                self._camera, prop_id, 0, ctypes.byref(data_type), ctypes.byref(data_size)
            ),
            f"EdsGetPropertySize(0x{prop_id:08X}) failed",
        )
        value = EdsUInt32()
        self._check(
            self._sdk.EdsGetPropertyData(self._camera, prop_id, 0, data_size, ctypes.byref(value)),
            f"EdsGetPropertyData(0x{prop_id:08X}) failed",
        )
        return value.value

    def _set_property_uint32(self, prop_id: int, value: int) -> None:
        assert self._sdk is not None
        v = EdsUInt32(value)
        self._check(
            self._sdk.EdsSetPropertyData(self._camera, prop_id, 0, ctypes.sizeof(v), ctypes.byref(v)),
            f"EdsSetPropertyData(0x{prop_id:08X}, 0x{value:08X}) failed",
        )

    def get_ae_mode(self) -> int:
        """Current shooting-mode dial position (kEdsPropID_AEMode). Read-only
        on cameras with a physical mode dial -- see kEdsAEMode_Manual."""
        self._require_session()
        return self._get_property_uint32(kEdsPropID_AEMode)

    def assert_manual_mode(self) -> None:
        """kEdsPropID_AEMode cannot be set via software on cameras with a
        physical mode dial (both the 1000D and T2i qualify) -- the dial has
        to already be turned to M by hand, or Av/Tv writes below may be
        silently overridden by the camera's own auto-exposure logic. Fail
        loud here rather than let that happen quietly mid-run."""
        mode = self.get_ae_mode()
        if mode != kEdsAEMode_Manual:
            raise CanonEDSDKError(
                f"Camera is not in Manual (M) exposure mode (AEMode=0x{mode:02X}). "
                f"Turn the physical mode dial to M before setting ISO/Av/Tv -- "
                f"this cannot be done via software on this camera."
            )

    def get_iso(self) -> int:
        self._require_session()
        return self._get_property_uint32(kEdsPropID_ISOSpeed)

    def set_iso(self, value: int) -> None:
        """value is the EDSDK hex code (e.g. 0x60 = ISO 800), not the ISO
        number itself -- see the ISO table in the EDSDK manual, section 5.2.22."""
        self._require_session()
        self._set_property_uint32(kEdsPropID_ISOSpeed, value)
        if self.verbose:
            print(f"[CanonEDSDK] ISO set to code 0x{value:08X}")

    def get_av(self) -> int:
        self._require_session()
        return self._get_property_uint32(kEdsPropID_Av)

    def set_av(self, value: int) -> None:
        """value is the EDSDK hex code (e.g. 0x30 = f/5.6), not the f-number
        itself -- see the Av table in the EDSDK manual, section 5.2.25."""
        self._require_session()
        self._set_property_uint32(kEdsPropID_Av, value)
        if self.verbose:
            print(f"[CanonEDSDK] Av set to code 0x{value:08X}")

    def get_tv(self) -> int:
        self._require_session()
        return self._get_property_uint32(kEdsPropID_Tv)

    def set_tv(self, value: int) -> None:
        """value is the EDSDK hex code (e.g. 0x64 = 1/45s), not the shutter
        speed itself -- see the Tv table in the EDSDK manual, section 5.2.26.
        Bulb cannot be set via software (Canon's own restriction)."""
        self._require_session()
        self._set_property_uint32(kEdsPropID_Tv, value)
        if self.verbose:
            print(f"[CanonEDSDK] Tv set to code 0x{value:08X}")

    def _set_unlimited_capacity(self) -> None:
        assert self._sdk is not None
        if not hasattr(self._sdk, 'EdsSetCapacity'):
            raise CanonEDSDKError("This EDSDK DLL does not export EdsSetCapacity")

        cap = EdsCapacity(
            numberOfFreeClusters=0x7FFFFFFF,
            bytesPerSector=512,
            reset=1,
        )
        self._check(self._sdk.EdsSetCapacity(self._camera, cap), "EdsSetCapacity failed")
        if self.verbose:
            print("[CanonEDSDK] Capacity set for host save")

    def _require_initialized(self) -> None:
        if not self._initialized or self._sdk is None:
            raise CanonEDSDKError("Camera SDK not initialized")

    def _require_session(self) -> None:
        self._require_initialized()
        if not self._session_open:
            raise CanonEDSDKError("Camera session is not open")

    @staticmethod
    def _check(code: int, message: str) -> None:
        if code != EDS_ERR_OK:
            raise CanonEDSDKError(message, code)


if __name__ == "__main__":
    # Minimal smoke-test script for Windows.
    # Update the DLL path before running.
    camera = CanonEDSDKCamera(
        dll_path="path/to/EDSDK.dll",
        save_dir="path/to/canon_test",
        auto_set_save_to_host=False,
        auto_set_capacity=False,
        verbose=True,
    )
    try:
        camera.initialize()
        camera.open_session()
        result = camera.capture_image(timeout_s=15.0)
        print(f"Capture saved to {result.path} in {result.elapsed_s:.3f}s")
    finally:
        camera.close()