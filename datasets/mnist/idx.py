import os
import gzip
import numpy as np
import struct


class IdxFile:
    """
    Optimized reader for IDX file formats (e.g., MNIST).
    """
    
    # Map IDX type codes to NumPy dtypes
    # 0x08: unsigned byte, 0x09: signed byte, etc.
    DATA_TYPES = {
        '08': 'uint8',
        '09': 'int8',
        '0b': 'int16',
        '0c': 'int32',
        '0d': 'float32',
        '0e': 'float64'
    }

    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Invalid file path: {filepath}")
        
        self.filepath = filepath
        self._open_file()
        self._read_header()

    def _open_file(self):
        if self.filepath.endswith('.gz'):
            self.f = gzip.open(self.filepath, 'rb')
        else:
            self.f = open(self.filepath, 'rb')

    def _read_header(self):
        # Read the first 4 bytes (Magic Number)
        # 0, 0, type_code, n_dims
        header = self.f.read(4)
        if len(header) < 4:
            raise ValueError("File too short to be a valid IDX file.")

        type_hex = header[2:3].hex()
        self.dtype = self.DATA_TYPES.get(type_hex)
        if not self.dtype:
            raise ValueError(f"Unsupported IDX data type: 0x{type_hex}")

        num_dims = header[3]

        # Read dimensions (each is a 4-byte big-endian int)
        # '>I' means big-endian unsigned int
        shape_data = self.f.read(4 * num_dims)
        self.shape = struct.unpack(f'>{num_dims}I', shape_data)
    
        # Calculate where the actual data starts
        self.header_offset = 4 + (4 * num_dims)
        self.item_size = np.dtype(self.dtype).itemsize

    def get_numpy_dtype(self):
        # Only add '>' for types larger than 1 byte
        if np.dtype(self.dtype).itemsize > 1:
            return f'>{self.dtype}'
        return self.dtype

    def read_all(self):
        self.f.seek(self.header_offset)
        target_dtype = self.get_numpy_dtype()
        
        # Read the buffer and cast to a standard platform-native type at the end
        data = np.frombuffer(self.f.read(), dtype=target_dtype)
        # return data.reshape(self.shape).astype(self.dtype)
        return data.reshape(self.shape).astype(self.dtype).copy()


    def read_at(self, idx) -> np.ndarray:
        if idx >= self.shape[0]:
            raise IndexError("Index out of range")

        # Items per entry (e.g., 28 * 28 = 784 for MNIST)
        items_per_entry = np.prod(self.shape[1:])
        # Total bytes per entry
        bytes_per_entry = int(items_per_entry * self.item_size)
        
        # Seek: skip header + (index * entry size)
        offset = self.header_offset + (idx * bytes_per_entry)
        self.f.seek(offset)
        
        target_dtype = self.get_numpy_dtype()
        raw_data = self.f.read(bytes_per_entry)
        
        return np.frombuffer(raw_data, dtype=target_dtype).reshape(self.shape[1:]).astype(self.dtype)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    with IdxFile('train-images-idx3-ubyte.gz') as idx:
        print(f"Data Shape: {idx.shape}")
        # Get everything
        all_images = idx.read_all()
        # Or just get the first image
        first_image = idx.read_at(0)

    print(all_images.shape)
    print(first_image.shape)