import struct
import array


# ------------------------------------------------------------------ #
# Tiny binary serialiser: writes lists of floats / ints to file
# ------------------------------------------------------------------ #
MAGIC = b"IOTIDS\x01"


def save(obj, path):
    """Serialise a dict of {str: list[float|int]}."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        _write_obj(f, obj)


def load(path):
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        assert magic == MAGIC, "Bad magic bytes"
        return _read_obj(f)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #
_TYPE_DICT  = 0
_TYPE_LIST  = 1
_TYPE_FLOAT = 2
_TYPE_INT   = 3
_TYPE_STR   = 4
_TYPE_NONE  = 5
_TYPE_BOOL  = 6
_TYPE_FARR  = 7   # array.array of floats


def _write_obj(f, obj):
    if obj is None:
        f.write(bytes([_TYPE_NONE]))
    elif isinstance(obj, bool):
        f.write(bytes([_TYPE_BOOL, int(obj)]))
    elif isinstance(obj, int):
        f.write(bytes([_TYPE_INT]))
        f.write(struct.pack(">q", obj))
    elif isinstance(obj, float):
        f.write(bytes([_TYPE_FLOAT]))
        f.write(struct.pack(">f", obj))
    elif isinstance(obj, str):
        enc = obj.encode("utf-8")
        f.write(bytes([_TYPE_STR]))
        f.write(struct.pack(">I", len(enc)))
        f.write(enc)
    elif isinstance(obj, array.array) and obj.typecode == "f":
        f.write(bytes([_TYPE_FARR]))
        f.write(struct.pack(">I", len(obj)))
        f.write(obj.tobytes())
    elif isinstance(obj, (list, tuple)):
        f.write(bytes([_TYPE_LIST]))
        f.write(struct.pack(">I", len(obj)))
        for item in obj:
            _write_obj(f, item)
    elif isinstance(obj, dict):
        f.write(bytes([_TYPE_DICT]))
        f.write(struct.pack(">I", len(obj)))
        for k, v in obj.items():
            _write_obj(f, k)
            _write_obj(f, v)
    else:
        raise TypeError(f"Unserialisable type: {type(obj)}")


def _read_obj(f):
    tag = ord(f.read(1))
    if tag == _TYPE_NONE:
        return None
    if tag == _TYPE_BOOL:
        return bool(ord(f.read(1)))
    if tag == _TYPE_INT:
        return struct.unpack(">q", f.read(8))[0]
    if tag == _TYPE_FLOAT:
        return struct.unpack(">f", f.read(4))[0]
    if tag == _TYPE_STR:
        n = struct.unpack(">I", f.read(4))[0]
        return f.read(n).decode("utf-8")
    if tag == _TYPE_FARR:
        n = struct.unpack(">I", f.read(4))[0]
        a = array.array("f")
        a.frombytes(f.read(n * 4))
        return a
    if tag == _TYPE_LIST:
        n = struct.unpack(">I", f.read(4))[0]
        return [_read_obj(f) for _ in range(n)]
    if tag == _TYPE_DICT:
        n = struct.unpack(">I", f.read(4))[0]
        return {_read_obj(f): _read_obj(f) for _ in range(n)}
    raise ValueError(f"Unknown tag {tag}")
