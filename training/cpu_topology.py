import ctypes
import os
import platform
from ctypes import wintypes


def _read_u16(raw, offset):
    return int.from_bytes(raw[offset : offset + 2], "little")


def _read_u32(raw, offset):
    return int.from_bytes(raw[offset : offset + 4], "little")


def windows_cpu_sets():
    if os.name != "nt":
        return []

    try:
        fn = ctypes.windll.kernel32.GetSystemCpuSetInformation
    except AttributeError:
        return []

    fn.argtypes = [
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.HANDLE,
        wintypes.ULONG,
    ]
    fn.restype = wintypes.BOOL

    returned = wintypes.ULONG(0)
    fn(None, 0, ctypes.byref(returned), None, 0)
    if returned.value <= 0:
        return []

    buffer = ctypes.create_string_buffer(returned.value)
    if not fn(buffer, returned, ctypes.byref(returned), None, 0):
        return []

    cpu_sets = []
    offset = 0
    raw = buffer.raw
    while offset + 32 <= returned.value:
        size = _read_u32(raw, offset)
        info_type = _read_u32(raw, offset + 4)
        if size <= 0:
            break

        if info_type == 0 and size >= 32:
            flags = raw[offset + 19]
            cpu_sets.append(
                {
                    "id": _read_u32(raw, offset + 8),
                    "group": _read_u16(raw, offset + 12),
                    "logical_processor": raw[offset + 14],
                    "core": raw[offset + 15],
                    "last_level_cache": raw[offset + 16],
                    "numa_node": raw[offset + 17],
                    "efficiency_class": raw[offset + 18],
                    "parked": bool(flags & 0x01),
                }
            )

        offset += size

    return sorted(
        cpu_sets,
        key=lambda item: (item["group"], item["logical_processor"]),
    )


def detect_cpu_topology():
    cpu_sets = windows_cpu_sets()
    total_logical = os.cpu_count() or 1
    topology = {
        "platform": platform.system(),
        "processor": platform.processor() or "unknown",
        "total_logical": total_logical,
        "cpu_sets": cpu_sets,
        "hybrid": False,
        "performance_logical": [],
        "efficient_logical": [],
        "parked_logical": [],
    }

    if not cpu_sets:
        topology["all_logical"] = list(range(total_logical))
        return topology

    classes = sorted({item["efficiency_class"] for item in cpu_sets})
    max_class = max(classes) if classes else 0
    topology["hybrid"] = len(classes) > 1
    topology["all_logical"] = [item["logical_processor"] for item in cpu_sets]
    topology["performance_logical"] = [
        item["logical_processor"]
        for item in cpu_sets
        if item["efficiency_class"] == max_class
    ]
    topology["efficient_logical"] = [
        item["logical_processor"]
        for item in cpu_sets
        if item["efficiency_class"] < max_class
    ]
    topology["parked_logical"] = [
        item["logical_processor"] for item in cpu_sets if item["parked"]
    ]
    return topology


def normalize_cpu_policy(value):
    normalized = str(value or "all_cores").strip().lower().replace("-", "_")
    aliases = {
        "auto": "all_cores",
        "all": "all_cores",
        "max": "all_cores",
        "allcores": "all_cores",
        "p": "performance_cores",
        "p_core": "performance_cores",
        "p_cores": "performance_cores",
        "performance": "performance_cores",
        "performance_core": "performance_cores",
        "e": "efficient_cores",
        "e_core": "efficient_cores",
        "e_cores": "efficient_cores",
        "efficient": "efficient_cores",
        "efficient_core": "efficient_cores",
    }
    return aliases.get(normalized, normalized)


def select_logical_processors(topology, policy):
    policy = normalize_cpu_policy(policy)
    all_logical = topology.get("all_logical") or list(range(topology["total_logical"]))

    if policy == "performance_cores" and topology.get("performance_logical"):
        return topology["performance_logical"], policy
    if policy == "efficient_cores" and topology.get("efficient_logical"):
        return topology["efficient_logical"], policy
    if policy == "balanced":
        performance = topology.get("performance_logical") or []
        efficient = topology.get("efficient_logical") or []
        if performance and efficient:
            efficient_budget = max(1, len(efficient) // 2)
            return performance + efficient[:efficient_budget], policy

    return all_logical, "all_cores"


def set_process_affinity(logical_processors):
    if os.name != "nt":
        return False

    logical_processors = sorted(set(int(item) for item in logical_processors))
    if not logical_processors or max(logical_processors) >= 64:
        return False

    mask = 0
    for logical_processor in logical_processors:
        mask |= 1 << logical_processor

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    return bool(kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask))


def set_process_priority(priority):
    if os.name != "nt":
        return False

    priority = str(priority or "normal").strip().lower()
    classes = {
        "idle": 0x00000040,
        "below_normal": 0x00004000,
        "normal": 0x00000020,
        "above_normal": 0x00008000,
        "high": 0x00000080,
    }
    priority_class = classes.get(priority)
    if priority_class is None:
        return False

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class))


def summarize_cpu_topology(topology, selected_logical, policy, affinity_applied, priority):
    perf = topology.get("performance_logical") or []
    eff = topology.get("efficient_logical") or []
    parked = topology.get("parked_logical") or []
    return {
        "processor": topology.get("processor", "unknown"),
        "hybrid": bool(topology.get("hybrid")),
        "policy": policy,
        "selected_logical": list(selected_logical),
        "selected_count": len(selected_logical),
        "total_logical": int(topology.get("total_logical", len(selected_logical))),
        "performance_count": len(perf),
        "efficient_count": len(eff),
        "parked_count": len(parked),
        "affinity_applied": bool(affinity_applied),
        "priority": priority,
    }
