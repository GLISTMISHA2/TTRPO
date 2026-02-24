import ctypes
import sys
from ctypes import c_float, c_double, c_int, POINTER, Structure

def call_or_fail(name, fn, args, expects_return):
    try:
        fn(*args)
        return True
    except:
        return False


class complex_float(Structure):
    _fields_ = [("real", c_float), ("imag", c_float)]


try:
    lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libopenblas.so")
except:
    print("Не удалось загрузить библиотеку")
    sys.exit(1)


def test_sasum():
    name = "cblas_sasum"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1], True)

def test_dasum():
    name = "cblas_dasum"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1], True)

def test_saxpy():
    name = "cblas_saxpy"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    y = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, c_float(1.0), x, 1, y, 1], False)

def test_daxpy():
    name = "cblas_daxpy"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    y = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, c_double(1.0), x, 1, y, 1], False)

def test_scopy():
    name = "cblas_scopy"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    y = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], False)

def test_dcopy():
    name = "cblas_dcopy"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    y = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], False)

def test_sdot():
    name = "cblas_sdot"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    y = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], True)

def test_ddot():
    name = "cblas_ddot"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    y = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], True)

def test_cdotc():
    name = "cblas_cdotc_sub"
    try:
        fn = getattr(lib, name)
        x = (complex_float * 1)()
        y = (complex_float * 1)()
        result = complex_float()
        fn(1, x, 1, y, 1, ctypes.byref(result))
        return True
    except:
        return False

def test_cdotu():
    name = "cblas_cdotu_sub"
    try:
        fn = getattr(lib, name)
        x = (complex_float * 1)()
        y = (complex_float * 1)()
        result = complex_float()
        fn(1, x, 1, y, 1, ctypes.byref(result))
        return True
    except:
        return False

def test_isamax():
    name = "cblas_isamax"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [1, x, 1,], True)

def test_idamax():
    name = "cblas_idamax"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [1, x, 1], True)

def test_snrm2():
    name = "cblas_snrm2"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1], True)

def test_dnrm2():
    name = "cblas_dnrm2"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1], True)

def test_sscal():
    name = "cblas_sscal"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, c_float(2.0), x, 1], False)

def test_dscal():
    name = "cblas_dscal"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, c_double(2.0), x, 1], False)

def test_sswap():
    name = "cblas_sswap"
    fn = getattr(lib, name)
    x = (c_float * 1)(0.0)
    y = (c_float * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], False)

def test_dswap():
    name = "cblas_dswap"
    fn = getattr(lib, name)
    x = (c_double * 1)(0.0)
    y = (c_double * 1)(0.0)
    return call_or_fail(name, fn, [0, x, 1, y, 1], False)

def test_srot():
    try:
        name = "cblas_srot"
        fn = getattr(lib, name)
        x = (c_float * 1)(0.0)
        y = (c_float * 1)(0.0)
        c = c_float(0.0)
        s = c_float(1.0)
        return call_or_fail(name, fn, [0, x, 1, y, 1, c, s], False)
    except:
        return False

def test_drot():
    try:
        name = "cblas_drot"
        fn = getattr(lib, name)
        x = (c_double * 1)(0.0)
        y = (c_double * 1)(0.0)
        c = c_double(0.0)
        s = c_double(1.0)
        return call_or_fail(name, fn, [0, x, 1, y, 1, c, s], False)
    except:
        return False

def test_srotg():
    try:
        name = "cblas_srotg"
        fn = getattr(lib, name)
        a = (c_float * 1)(1.0)
        b = (c_float * 1)(1.0)
        c = (c_float * 1)(0.0)
        s = (c_float * 1)(0.0)
        return call_or_fail(name, fn, [a, b, c, s], False)
    except:
        return False

def test_drotg():
    try:
        name = "cblas_drotg"
        fn = getattr(lib, name)
        a = (c_double * 1)(1.0)
        b = (c_double * 1)(1.0)
        c = (c_double * 1)(0.0)
        s = (c_double * 1)(0.0)
        return call_or_fail(name, fn, [a, b, c, s], False)
    except:
        return False


tests = [
    ("sasum", test_sasum),
    ("dasum", test_dasum),
    ("saxpy", test_saxpy),
    ("daxpy", test_daxpy),
    ("scopy", test_scopy),
    ("dcopy", test_dcopy),
    ("sdot", test_sdot),
    ("ddot", test_ddot),
    ("cdotc", test_cdotc),
    ("cdotu", test_cdotu),
    ("isamax", test_isamax),
    ("idamax", test_idamax),
    ("snrm2", test_snrm2),
    ("dnrm2", test_dnrm2),
    ("sscal", test_sscal),
    ("dscal", test_dscal),
    ("sswap", test_sswap),
    ("dswap", test_dswap),
    ("srot", test_srot),
    ("drot", test_drot),
    ("srotg", test_srotg),
    ("drotg", test_drotg),
]

def main():
    results = {}
    failed = []
    
    for name, test_func in tests:
        results[name] = test_func()
        if not results[name]:
            failed.append(name)
    
    print("\nРЕЗУЛЬТАТЫ:\n")
    for name, passed in results.items():
        status = "Пройден" if passed else "НЕ пройден"
        print(f"{status} {name}")
    

    if not failed:
        print(" ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print(f" НЕ ПРОЙДЕНЫ: {', '.join(failed)}")

if __name__ == "__main__":
    main()