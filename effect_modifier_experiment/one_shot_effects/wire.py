"""Deterministic, data-only wire encoding with honest serialized byte counts."""
import hashlib
import json
import struct

import numpy as np

MAGIC = b'FFEMDR1\n'
BUDGET_MAGIC = b'FFBUDG1\n'
DTYPES = {'<f8', '<f4', '<i8', '<i4', '|u1'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate wire metadata key')
        value[key] = item
    return value


def pack_payload(method, arrays, metadata):
    if not isinstance(method, str) or not method or not isinstance(metadata, dict):
        raise ValueError('invalid wire identity/metadata')
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError('named numerical arrays required')
    descriptors, chunks, offset = {}, [], 0
    for name in sorted(arrays):
        if not isinstance(name, str) or not name or name.startswith('_'):
            raise ValueError('invalid wire array name')
        a = np.asarray(arrays[name])
        dtype = a.dtype.newbyteorder('<')
        if dtype.str not in DTYPES or not np.isfinite(a).all():
            raise ValueError('unsupported dtype or nonfinite payload array')
        a = np.array(a, dtype=dtype, order='C', copy=True)
        blob = a.tobytes()
        descriptors[name] = dict(shape=list(a.shape), dtype=a.dtype.str,
                                 offset=offset, nbytes=len(blob))
        chunks.append(blob)
        offset += len(blob)
    header = canonical(dict(wire_version=1, method=method, metadata=metadata,
                            arrays=descriptors)).encode('utf-8')
    return MAGIC + struct.pack('<Q', len(header)) + header + b''.join(chunks)


def unpack_payload(blob, expected_method=None):
    if isinstance(blob, bytes) and blob.startswith(BUDGET_MAGIC):
        blob = unpad_payload(blob)
    if not isinstance(blob, bytes) or not blob.startswith(MAGIC) or len(blob) < len(MAGIC)+8:
        raise ValueError('invalid serialized payload')
    length = struct.unpack('<Q', blob[len(MAGIC):len(MAGIC)+8])[0]
    start = len(MAGIC)+8
    if length > len(blob)-start:
        raise ValueError('truncated wire metadata')
    header = json.loads(blob[start:start+length].decode('utf-8'), object_pairs_hook=_unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
    if (set(header) != {'wire_version','method','metadata','arrays'}
            or type(header['wire_version']) is not int or header['wire_version'] != 1
            or not isinstance(header['method'],str) or not header['method']):
        raise ValueError('wire contract mismatch')
    if expected_method is not None and header['method'] != expected_method:
        raise ValueError('wrong method payload')
    if not isinstance(header['metadata'], dict) or not isinstance(header['arrays'], dict):
        raise ValueError('invalid wire metadata types')
    content, offset, arrays = memoryview(blob)[start+length:], 0, {}
    for name, item in sorted(header['arrays'].items()):
        if set(item) != {'shape','dtype','offset','nbytes'} or item['dtype'] not in DTYPES:
            raise ValueError('invalid wire array descriptor')
        shape = item['shape']
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid wire array shape')
        size = int(np.prod(shape, dtype=object)) * np.dtype(item['dtype']).itemsize
        if (type(item['offset']) is not int or type(item['nbytes']) is not int
                or item['offset'] != offset or item['nbytes'] != size or offset+size > len(content)):
            raise ValueError('invalid wire array bounds')
        a = np.frombuffer(content[offset:offset+size], dtype=item['dtype']).reshape(shape).copy()
        if not np.isfinite(a).all():
            raise ValueError('nonfinite received payload')
        a.setflags(write=False)
        arrays[name], offset = a, offset+size
    if not arrays or offset != len(content):
        raise ValueError('empty payload or trailing wire bytes')
    return arrays, header['metadata']


def pad_payload(blob, budget_bytes):
    """Transparent fixed-byte diagnostic envelope; padding is charged in full.

    Equal-rank strategies have nearly equal array costs but unequal JSON header
    lengths. This envelope supports exact equal-wire-byte comparisons without
    claiming that padding improves compression or changing the fitted model.
    A deployed fixed budget must be distributed before local construction.
    """
    unpack_payload(blob)
    overhead = len(BUDGET_MAGIC)+8
    if type(budget_bytes) is not int or budget_bytes < len(blob)+overhead:
        raise ValueError('payload does not fit the declared byte budget')
    return BUDGET_MAGIC+struct.pack('<Q',len(blob))+blob+bytes(budget_bytes-len(blob)-overhead)


def unpad_payload(blob):
    start = len(BUDGET_MAGIC)+8
    if not isinstance(blob,bytes) or not blob.startswith(BUDGET_MAGIC) or len(blob)<start:
        raise ValueError('invalid byte-budget envelope')
    size = struct.unpack('<Q',blob[len(BUDGET_MAGIC):start])[0]
    if size>len(blob)-start or any(blob[start+size:]):
        raise ValueError('invalid byte-budget padding')
    content = blob[start:start+size]
    if not content.startswith(MAGIC):
        raise ValueError('nested/invalid byte-budget content')
    return content


def communication_report(payloads, protocol_bytes=0, model_broadcast_bytes=0):
    if not payloads or any(not isinstance(p, bytes) for p in payloads):
        raise ValueError('actual serialized site payloads required')
    if any(type(n) is not int or n < 0 for n in (protocol_bytes,model_broadcast_bytes)):
        raise ValueError('invalid separate distribution byte counts')
    return dict(upload_rounds=1, upload_messages=len(payloads),
                upload_bytes=sum(len(p) for p in payloads),
                per_site_upload_bytes=[len(p) for p in payloads],
                payload_sha256=[hashlib.sha256(p).hexdigest() for p in payloads],
                protocol_distribution_bytes=protocol_bytes,
                final_model_distribution_bytes=model_broadcast_bytes,
                network_time_measured=False,
                encoding='FFEMDR1 canonical JSON header and little-endian raw arrays')
