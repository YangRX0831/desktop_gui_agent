"""提供 Windows 系统主音量的只读感知。

职责：
    通过 IAudioEndpointVolume COM 接口读取默认渲染设备的标量主音量，
    供 Agent 主循环注入 Prompt 动态状态；模型据此计算精确音量调整。

安全边界：
    只读查询，不修改音量、不写入日志正文。任何 COM 或查询失败都返回
    None，由调用方按 unknown 处理，不猜测音量值。
"""

import ctypes
import uuid
from ctypes import HRESULT, POINTER, WINFUNCTYPE, byref, c_float, c_uint, c_void_p

ole32 = ctypes.windll.ole32

CLSID_MMDEVICE_ENUMERATOR = uuid.UUID(
    "BCDE0395-E52F-467C-8E3D-C4579291692E",
).bytes_le
IID_IMM_DEVICE_ENUMERATOR = uuid.UUID(
    "A95664D2-9614-4F35-A746-DE8DB63617E6",
).bytes_le
IID_IAUDIO_ENDPOINT_VOLUME = uuid.UUID(
    "5CDF2C82-841E-4546-9722-0CF74078229A",
).bytes_le


def _vtable(obj: int) -> list:
    """读取 COM 对象的函数表;失败抛出由调用方统一处理。"""
    import ctypes as ct

    return ct.cast(  # type: ignore[return-value]  # ctypes 指针链无存根
        ct.cast(obj, POINTER(c_void_p))[0],
        POINTER(c_void_p),
    )


def _release_interface(pointer) -> None:
    """对非空 COM 接口指针调用 IUnknown::Release(vtable[2])。

    best-effort 清理:失败静默返回,不掩盖主结果或主异常。
    """
    if not pointer.value:
        return
    try:
        release = WINFUNCTYPE(c_uint, c_void_p)(_vtable(pointer.value)[2])
        release(pointer)
    except Exception:
        return


def get_master_volume_percent() -> int | None:
    """返回当前系统主音量百分比(0-100);不可用时返回 None。"""
    try:
        hr_init = ole32.CoInitializeEx(None, 4)
    except Exception:
        return None
    # 失败 HRESULT(如 RPC_E_CHANGED_MODE)表示 COM 未初始化,不得配对卸载。
    if hr_init < 0:
        return None
    enumerator = c_void_p()
    device = c_void_p()
    endpoint = c_void_p()
    try:
        hr = ole32.CoCreateInstance(
            CLSID_MMDEVICE_ENUMERATOR,
            None,
            23,
            IID_IMM_DEVICE_ENUMERATOR,
            byref(enumerator),
        )
        if hr != 0 or not enumerator.value:
            return None

        get_default = WINFUNCTYPE(
            HRESULT,
            c_void_p,
            c_uint,
            c_uint,
            POINTER(c_void_p),
        )(_vtable(enumerator.value)[4])
        hr = get_default(enumerator, 0, 1, byref(device))
        if hr != 0 or not device.value:
            return None

        activate = WINFUNCTYPE(
            HRESULT,
            c_void_p,
            c_void_p,
            c_uint,
            c_void_p,
            POINTER(c_void_p),
        )(_vtable(device.value)[3])
        hr = activate(
            device,
            IID_IAUDIO_ENDPOINT_VOLUME,
            23,
            None,
            byref(endpoint),
        )
        if hr != 0 or not endpoint.value:
            return None

        get_level = WINFUNCTYPE(
            HRESULT,
            c_void_p,
            POINTER(c_float),
        )(_vtable(endpoint.value)[9])
        level = c_float()
        hr = get_level(endpoint, byref(level))
        if hr != 0:
            return None
        return round(level.value * 100)
    except Exception:
        return None
    finally:
        # 逆序释放已获取接口,再配对卸载 COM;清理失败不掩盖主结果。
        for pointer in (endpoint, device, enumerator):
            _release_interface(pointer)
        try:
            ole32.CoUninitialize()
        except Exception:
            # best-effort 卸载;失败不影响已计算的主结果或主异常。
            pass
