# -*- coding: utf-8 -*-
"""
LeetGPU 解题通用本地验证器。

干嘛的：把任意 solution.cu 的「纯 device 代码」用 NVRTC 编成 PTX，
再用 CUDA Driver API 在本机真卡上跑起来，配合 numpy 参考值做校验。

为什么这么绕：本机 cl.exe 启动会调被安全策略拉黑的 reg.exe，nvcc 不可用。
NVRTC 只编 device 代码（不需要 host 编译器），host 侧分派逻辑用 Python 复刻。

典型用法（在每题的 check.py 里）：

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_tools"))
    from localrun import Runner, nvrtc_compile, find_entry_names

    r = Runner.from_file("solution.cu")
    din = r.to_device(x)          # numpy -> device
    dout = r.alloc(y.nbytes)
    r.launch("vec_add", grid, block, [din, dout, r.i(N)])
    r.sync()
    out = r.from_device(dout, y.shape, y.dtype)
    assert np.allclose(out, ref, atol=1e-5)
"""
import ctypes
import os
import re
import sys
from ctypes import (c_char_p, c_float, c_int, c_longlong, c_double, c_size_t,
                    c_uint, c_ubyte, c_void_p, byref, cast, create_string_buffer, POINTER)

import numpy as np

_CUDA_BIN_CANDIDATES = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
]


def _find_nvrtc():
    """挑 nvrtc 主库。

    坑：目录里同时有 nvrtc64_120_0.dll 和 nvrtc64_120_0.alt.dll，后者名字排序更前，
    naive 的 listdir+startswith 会挑到 .alt —— 而 .alt 要求配套的
    nvrtc-builtins.altXX_XXX.dll，通常并不存在，报
    "failed to open nvrtc-builtins.alt64_128.dll"。必须把 .alt 排除。
    """
    for d in _CUDA_BIN_CANDIDATES:
        if not os.path.isdir(d):
            continue
        cands = [f for f in sorted(os.listdir(d))
                 if f.startswith("nvrtc64_") and f.endswith(".dll")
                 and "builtins" not in f and ".alt" not in f]
        # 优先选有配套 builtins 的那个
        for f in cands:
            ver = f[len("nvrtc64_"):].split("_")[0]        # '120'
            builtin = "nvrtc-builtins64_%s.dll" % ver
            if os.path.exists(os.path.join(d, builtin)):
                return os.path.join(d, f)
        if cands:
            return os.path.join(d, cands[0])
    raise RuntimeError("找不到 nvrtc64_*.dll")


NVRTC_DLL = _find_nvrtc()
CUDA_BIN = os.path.dirname(NVRTC_DLL)
# nvrtc-builtins.dll 要能被找到
os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

nvrtc = ctypes.WinDLL(NVRTC_DLL)
nvrtc.nvrtcCreateProgram.restype = c_int
nvrtc.nvrtcCreateProgram.argtypes = [POINTER(c_void_p), c_char_p, c_char_p,
                                     c_int, POINTER(c_char_p), POINTER(c_char_p)]
nvrtc.nvrtcCompileProgram.argtypes = [c_void_p, c_int, POINTER(c_char_p)]
nvrtc.nvrtcCompileProgram.restype = c_int
nvrtc.nvrtcGetPTXSize.argtypes = [c_void_p, POINTER(c_size_t)]
nvrtc.nvrtcGetPTXSize.restype = c_int
nvrtc.nvrtcGetPTX.argtypes = [c_void_p, c_char_p]
nvrtc.nvrtcGetPTX.restype = c_int
nvrtc.nvrtcGetProgramLogSize.argtypes = [c_void_p, POINTER(c_size_t)]
nvrtc.nvrtcGetProgramLogSize.restype = c_int
nvrtc.nvrtcGetProgramLog.argtypes = [c_void_p, c_char_p]
nvrtc.nvrtcGetProgramLog.restype = c_int

# CUDA 头文件目录（fp16 题要用 cuda_fp16.h；NVRTC 不自带头文件）
CUDA_INCLUDE_DIRS = []
for _b in _CUDA_BIN_CANDIDATES:
    _inc = os.path.join(os.path.dirname(_b), "include")
    if os.path.isdir(_inc) and _inc not in CUDA_INCLUDE_DIRS:
        CUDA_INCLUDE_DIRS.append(_inc)

DEFAULT_ARCHS = ["compute_120", "compute_100", "compute_90", "compute_80"]


class NvrtcError(RuntimeError):
    pass


def strip_host_code(src):
    """去掉 #include 与 host 入口 solve()，只留 device 代码给 NVRTC。

    正则必须行首锚定：文件顶部注释里常也写着 `extern "C" void solve(...)`，
    不锚定会被注释骗到，把整个 kernel 段全砍掉（症状：PTX 只有 0.2KB、0 符号）。
    """
    drop = ("cuda_runtime.h", "cuda_runtime_api.h", "math_constants.h", "stdint.h")
    lines = []
    for ln in src.splitlines():
        st = ln.strip()
        if st.startswith("#include") and any(d in st for d in drop):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    m = re.search(r'(?m)^\s*extern\s+"C"\s+void\s+solve\s*\(', text)
    if m:
        text = text[:m.start()]
    # 有些题把 solve 写成 __global__ 之外的包装，去掉 main()
    m2 = re.search(r"(?m)^\s*int\s+main\s*\(", text)
    if m2:
        text = text[:m2.start()]
    return text


def nvrtc_compile(src, name="kernel.cu", archs=None, extra_opts=()):
    """device 源码 -> (ptx, arch)。逐个 arch 试。"""
    archs = archs or DEFAULT_ARCHS
    logs = []
    for arch in archs:
        prog = c_void_p()
        opts = [("-arch=" + arch).encode(), b"-std=c++17", b"-default-device"]
        for inc in CUDA_INCLUDE_DIRS:
            opts.append(("-I" + inc).encode())
        opts += [o.encode() if isinstance(o, str) else o for o in extra_opts]
        arr = (c_char_p * (len(opts) + 1))()
        for i, o in enumerate(opts):
            arr[i] = o
        arr[len(opts)] = None

        if nvrtc.nvrtcCreateProgram(byref(prog), src.encode("utf-8"),
                                    name.encode("utf-8"), 0, None, None) != 0:
            raise NvrtcError("nvrtcCreateProgram 失败")
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)

        sz = c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, byref(sz))
        lb = create_string_buffer(sz.value or 1)
        nvrtc.nvrtcGetProgramLog(prog, lb)
        log = lb.value.decode("utf-8", "replace")
        if err != 0:
            logs.append("[%s] %s" % (arch, log.strip()[:500]))
            continue

        psz = c_size_t()
        if nvrtc.nvrtcGetPTXSize(prog, byref(psz)) != 0:
            continue
        pb = create_string_buffer(psz.value)
        if nvrtc.nvrtcGetPTX(prog, pb) != 0:
            continue
        return pb.value.decode("utf-8", "replace"), arch, log
    raise NvrtcError("所有 arch 编译失败:\n" + "\n".join(logs))


def find_entry_names(ptx):
    return re.findall(r"\.visible\s+\.entry\s+([A-Za-z0-9_$]+)\s*\(", ptx)


def _demangle_first(sym):
    """从 Itanium mangled 名里精确取出第一段（函数名本体）。

    规则：`_Z <len> <name> <其余类型编码>`，`len` 是十进制**字符数**，
    必须恰好截这么多个字符 —— 不能靠正则贪婪匹配（见 symbol_map 的注释）。

        _Z11gemm_wmma64PK6__halfS1_PS_iiiff  ->  'gemm_wmma64'   (len=11)
        _Z16gemm_wmma64_bk64PK6__half...     ->  'gemm_wmma64_bk64' (len=16)
    """
    m = re.match(r"_Z(\d+)", sym)
    if not m:
        return None
    n = int(m.group(1))
    name = sym[m.end():m.end() + n]
    if len(name) != n or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return None
    return name



# ---------------------------------------------------------------- Driver API
cuda = ctypes.WinDLL("nvcuda.dll")


def _bind(name, argtypes):
    f = getattr(cuda, name)
    f.argtypes = argtypes
    f.restype = c_int
    # ctypes 的 CDLL.__getattr__ 每次都会**新建**一个 _FuncPtr，
    # 于是 `cuda.cuMemcpyHtoD` 这种写法拿到的是没设过 argtypes 的新对象，
    # 64 位指针会被当 C int 截断 -> 报 "OverflowError: int too long to convert"，
    # 或静默写到错误地址（表现为 memset 返回 201 = 无效上下文）。
    # 把配置好的对象塞回实例字典缓存住，两种写法就都安全。
    try:
        setattr(cuda, name, f)
    except Exception:
        pass
    return f


cuInit = _bind("cuInit", [c_uint])
cuDeviceGet = _bind("cuDeviceGet", [POINTER(c_int), c_int])
cuDeviceGetName = _bind("cuDeviceGetName", [c_char_p, c_int, c_int])
cuDeviceGetAttribute = _bind("cuDeviceGetAttribute", [POINTER(c_int), c_int, c_int])
cuCtxCreate = _bind("cuCtxCreate_v2", [POINTER(c_void_p), c_uint, c_int])
cuModuleLoadData = _bind("cuModuleLoadData", [POINTER(c_void_p), c_void_p])
cuModuleGetFunction = _bind("cuModuleGetFunction", [POINTER(c_void_p), c_void_p, c_char_p])
cuMemAlloc = _bind("cuMemAlloc_v2", [POINTER(c_void_p), c_size_t])
cuMemFree = _bind("cuMemFree_v2", [c_void_p])
cuMemsetD8 = _bind("cuMemsetD8_v2", [c_void_p, c_ubyte, c_size_t])
cuMemsetD32 = _bind("cuMemsetD32_v2", [c_void_p, c_uint, c_size_t])
cuMemcpyHtoD = _bind("cuMemcpyHtoD_v2", [c_void_p, c_void_p, c_size_t])
cuMemcpyDtoH = _bind("cuMemcpyDtoH_v2", [c_void_p, c_void_p, c_size_t])
cuLaunchKernel = _bind("cuLaunchKernel",
                       [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                        c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p)])
cuCtxSynchronize = _bind("cuCtxSynchronize", [])
cuEventCreate = _bind("cuEventCreate", [POINTER(c_void_p), c_uint])
cuEventRecord = _bind("cuEventRecord", [c_void_p, c_void_p])
cuEventSynchronize = _bind("cuEventSynchronize", [c_void_p])
cuEventElapsedTime = _bind("cuEventElapsedTime", [POINTER(c_float), c_void_p, c_void_p])
cuEventDestroy = _bind("cuEventDestroy_v2", [c_void_p])
cuGetErrorString = _bind("cuGetErrorString", [c_int, POINTER(c_char_p)])


def cu_check(r, what=""):
    if r != 0:
        p = c_char_p()
        try:
            cuGetErrorString(r, byref(p))
            msg = (p.value or b"").decode("utf-8", "replace")
        except Exception:
            msg = str(r)
        raise RuntimeError("CUDA error @ %s: %s (%d)" % (what, msg, r))


class Runner:
    """装载 PTX 并 launch kernel。"""

    _ctx_ready = False
    _devname = ""

    def __init__(self, ptx, arch="", log=""):
        if not Runner._ctx_ready:
            cu_check(cuInit(0), "cuInit")
            dev = c_int()
            cu_check(cuDeviceGet(byref(dev), 0), "cuDeviceGet")
            nb = create_string_buffer(256)
            cuDeviceGetName(nb, 256, dev)
            Runner._devname = nb.value.decode("utf-8", "replace")
            ctx = c_void_p()
            cu_check(cuCtxCreate(byref(ctx), 0, dev), "cuCtxCreate")
            Runner._ctx_ready = True
        self.arch = arch
        self.log = log
        self.mod = c_void_p()
        buf = create_string_buffer(ptx.encode("utf-8"))
        cu_check(cuModuleLoadData(byref(self.mod), cast(buf, c_void_p)), "cuModuleLoadData")
        self.names = find_entry_names(ptx)
        self._fn = {}

    @classmethod
    def from_file(cls, path, **kw):
        src = open(path, encoding="utf-8").read()
        ptx, arch, log = nvrtc_compile(strip_host_code(src), name=os.path.basename(path), **kw)
        return cls(ptx, arch, log)

    @classmethod
    def devicename(cls):
        return Runner._devname

    # -- 内存 --
    def alloc(self, nbytes):
        p = c_void_p()
        cu_check(cuMemAlloc(byref(p), c_size_t(max(int(nbytes), 4))), "cuMemAlloc")
        return p

    def free(self, p):
        cuMemFree(p)

    def ptr_add(self, p, nbytes):
        """在 device 指针上做偏移（用于一个缓冲里切多段，镜像 solve() 的布局）。"""
        return c_void_p(p.value + int(nbytes))

    def to_device(self, arr):
        arr = np.ascontiguousarray(arr)
        p = self.alloc(arr.nbytes)
        cu_check(cuMemcpyHtoD(p, arr.ctypes.data, c_size_t(arr.nbytes)), "h2d")
        return p

    def from_device(self, p, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        cu_check(cuMemcpyDtoH(out.ctypes.data, p, c_size_t(out.nbytes)), "d2h")
        return out

    # -- 执行 --
    def symbol_map(self):
        """mangled 符号 -> 原始函数名。Itanium ABI: _Z<len><name>...

        ⚠️ 必须按**长度前缀**取字符，不能用贪婪正则。早先的写法是
        `_Z(\\d+)([A-Za-z_][A-Za-z0-9_]*)` —— 第二组会把
        `PK6__halfS1_PS_iiiff` 这种后续类型编码也一起吃掉（它们全是
        word 字符），于是 `len(group2) == int(group1)` 永远不成立，
        所有符号都退化成 `m[sym] = sym`，**精确查名彻底失效**，
        只能靠"唯一子串"兜底。一旦出现前缀包含关系（如
        `gemm_wmma64` 与 `gemm_wmma64_bk64`）就直接报"多个候选"。
        """
        if getattr(self, "_smap", None) is None:
            m = {}
            for sym in self.names:
                name = _demangle_first(sym)
                if name:
                    m[name] = sym
                else:
                    # 解不出来（非 _Z 前缀等）就按原样登记
                    m[sym] = sym
            self._smap = m
        return self._smap

    def func(self, want):
        """按**精确函数名**取 kernel。

        早先用子串匹配踩过坑：`"vec_add"` 会先命中 `vec_add1`（5 个参数），
        而调用方按 `vec_add4`（4 个参数）传参 -> 多余形参读到脏值 -> 越界。
        """
        if want not in self._fn:
            smap = self.symbol_map()
            mangled = smap.get(want)
            if mangled is None:
                # 退一步：允许唯一子串匹配，多个候选直接报错
                cands = [v for k, v in smap.items() if want in k]
                if len(cands) == 1:
                    mangled = cands[0]
                elif len(cands) > 1:
                    raise RuntimeError("kernel 名 %r 有多个候选: %s"
                                       % (want, [k for k in smap if want in k]))
            if mangled is None:
                raise RuntimeError("PTX 里找不到 kernel %r；实际有: %s"
                                   % (want, ", ".join(sorted(self.symbol_map()))))
            f = c_void_p()
            cu_check(cuModuleGetFunction(byref(f), self.mod, mangled.encode()),
                     "getFunc " + mangled)
            self._fn[want] = f
        return self._fn[want]

    def launch(self, kernel, grid, block, args, shared=0):
        """kernel: 精确函数名；grid/block: int 或 3 元组；args: ctypes 对象列表"""
        def xyz(v):
            if isinstance(v, (tuple, list)):
                a = list(v) + [1, 1, 1]
                return int(a[0]), int(a[1]), int(a[2])
            return int(v), 1, 1
        gx, gy, gz = xyz(grid)
        bx, by, bz = xyz(block)
        refs = [cast(byref(a), c_void_p) for a in args]
        arr = (c_void_p * len(refs))(*refs)
        f = self.func(kernel) if isinstance(kernel, str) else kernel
        cu_check(cuLaunchKernel(f, gx, gy, gz, bx, by, bz, int(shared),
                                None, arr, None), "launch " + str(kernel))

    def sync(self):
        cu_check(cuCtxSynchronize(), "sync")

    def check_err(self):
        """同步一次，把异步错误暴露出来。"""
        self.sync()

    # -- 便捷 --
    @staticmethod
    def i(v):
        return c_int(int(v))

    @staticmethod
    def ll(v):
        """long long 形参必须用这个 —— 用 c_int 会少 4 字节导致参数整体错位。"""
        return c_longlong(int(v))

    @staticmethod
    def d(v):
        return c_double(float(v))

    @staticmethod
    def f(v):
        return c_float(float(v))

    @staticmethod
    def p(v):
        return c_void_p(v) if not isinstance(v, c_void_p) else v

    def timeit(self, fn, repeat=50, warmup=10):
        for _ in range(warmup):
            fn()
        self.sync()
        best = None
        for _ in range(3):
            e0, e1 = c_void_p(), c_void_p()
            cuEventCreate(byref(e0), 0)
            cuEventCreate(byref(e1), 0)
            cuEventRecord(e0, None)
            for _ in range(repeat):
                fn()
            cuEventRecord(e1, None)
            cuEventSynchronize(e1)
            ms = c_float()
            cuEventElapsedTime(byref(ms), e0, e1)
            v = ms.value / repeat
            best = v if best is None else min(best, v)
            cuEventDestroy(e0)
            cuEventDestroy(e1)
        return best


# ------------------------------------------------------- 常见分派镜像
def run_vec_then_tail(r, name4, name1, ins, out, N, elems=4, block=256,
                      max_grid=65535, tail=True):
    """镜像 solve() 里最常见的分派：向量化主路径 + 标量尾部。

    和 solution.cu 的 solve() 保持一致（否则本地测的就不是真实路径）：
      对齐且 N>=4 -> name4 处理 n4 个向量单元，剩下的尾巴交给 name1
      否则        -> name1 全量处理
    """
    if N >= elems:
        n4 = N // elems
        g = min(max_grid, (n4 + block - 1) // block) or 1
        r.launch(name4, g, block, list(ins) + [out, r.i(n4)])
        start = n4 * elems
        if tail and start < N:
            r.launch(name1, 1, block, list(ins) + [out, r.i(N), r.i(start)])
    else:
        g = min(max_grid, (N + block - 1) // block) or 1
        r.launch(name1, g, block, list(ins) + [out, r.i(N), r.i(0)])


def run_simple(r, name, ins, out, N, block=256, max_grid=65535, out_index=-1):
    """单个 kernel 覆盖全量的常见分派：args = ins + [out] + [N]。

    out_index 不用；保留签名一致性。
    """
    g = min(max_grid, (N + block - 1) // block) or 1
    r.launch(name, g, block, list(ins) + [out, r.i(N)])


# ---------------------------------------------------------------- 断言工具
def compare(got, ref, name="output", rtol=1e-4, atol=1e-5):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        return False, "%s shape 不符: got %s vs ref %s" % (name, got.shape, ref.shape)
    if not got.size:
        return True, "%s OK (空)" % name
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    tol = atol + rtol * np.abs(ref.astype(np.float64))
    ok = bool(np.all(diff <= tol))
    if ok:
        return True, "%s OK (max|diff|=%.3e)" % (name, diff.max())
    # 报「最严重越界」的元素，而不是 max|diff| 的 —— 后者在 ref 很大时
    # 即使相对误差极小也会是最大值，指不到真正违规的位置。
    ratio = np.where(tol > 0, diff / tol, np.where(diff > 0, np.inf, 0.0))
    idx = int(np.argmax(ratio))
    return False, ("%s FAIL: diff=%.3e 超出容差 %.3e (超 %.1fx) @ %d  got=%s ref=%s"
                   % (name, diff.ravel()[idx], tol.ravel()[idx], ratio.ravel()[idx], idx,
                      got.ravel()[idx], ref.ravel()[idx]))
