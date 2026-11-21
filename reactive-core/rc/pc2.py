"""PointCloud2 <-> numpy (numpy only; copied from frontier-fastlio/dog_codes/common.py, not imported)."""
import numpy as np

_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
       7: np.float32, 8: np.float64}


def cloud_to_xyz(msg):
    """PointCloud2 -> (N,3) float32 x,y,z (finite points only)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    off = {f.name: f for f in msg.fields}
    if not all(k in off for k in 'xyz'):
        return np.zeros((0, 3), np.float32)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = msg.point_step
    buf = buf[:n * step].reshape(n, step)
    cols = []
    for k in 'xyz':
        f = off[k]
        dt = np.dtype(_DT[f.datatype]).newbyteorder('>' if msg.is_bigendian else '<')
        sz = dt.itemsize
        col = np.ascontiguousarray(buf[:, f.offset:f.offset + sz]).view(dt).reshape(n)
        cols.append(col.astype(np.float32))
    xyz = np.stack(cols, axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def xyz_to_cloud(xyz, frame_id, stamp):
    from sensor_msgs.msg import PointCloud2, PointField
    m = PointCloud2()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    m.height = 1
    m.width = int(xyz.shape[0])
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate('xyz')]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * m.width
    m.is_dense = True
    m.data = xyz.tobytes()
    return m
