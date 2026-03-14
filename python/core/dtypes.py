INT8   = "int8"
INT16  = "int16"
FLOAT16 = "float16"
FLOAT32 = "float32"

ITEMSIZE = {INT8: 1, INT16: 2, FLOAT16: 2, FLOAT32: 4}

INT8_MIN  = -128
INT8_MAX  =  127
INT8_RANGE = 255

def size_bytes(dtype, n_elements):
    return ITEMSIZE[dtype] * n_elements

def cast_val(v, dtype):
    if dtype == INT8:
        v = int(round(v))
        return max(INT8_MIN, min(INT8_MAX, v))
    if dtype == FLOAT32:
        return float(v)
    return v
