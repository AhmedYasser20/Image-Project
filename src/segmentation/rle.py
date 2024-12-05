import numpy as np

class RLE:
    """
    Class for Run-Length Encoding (RLE) operations.
    """

    @staticmethod
    def encode(array):
        """
        Encode a binary array using Run-Length Encoding (RLE).

        Parameters:
        array (numpy array): Input binary array.

        Returns:
        tuple: RLE lengths, values.
        """
        if len(array) == 0:
            return [], [], []

        copied_array = np.copy(array)
        first_dismatch = np.array(copied_array[1:] != copied_array[:-1])
        dismatch_positions = np.append(np.where(first_dismatch), len(copied_array) - 1)
        rle_lengths = np.diff(np.append(-1, dismatch_positions))
        rle_values = [copied_array[i] for i in np.cumsum(np.append(0, rle_lengths))[:-1]]
        return rle_lengths, rle_values

    @staticmethod
    def hv_encode(image, axis=1):
        """
        Encode a binary image using Run-Length Encoding (RLE) along specified axis.

        Parameters:
        image (numpy array): Input binary image.
        axis (int): Axis along which to perform RLE (0 for rows, 1 for columns).

        Returns:
        tuple: RLE lengths, values.
        """
        rle_lengths, rle_values = [], []

        if axis == 1:
            for i in range(image.shape[1]):
                col_rle_lengths, col_rle_values = RLE.encode(image[:, i])
                rle_lengths.append(col_rle_lengths)
                rle_values.append(col_rle_values)
        else:
            for i in range(image.shape[0]):
                row_rle_lengths, row_rle_values = RLE.encode(image[i])
                rle_lengths.append(row_rle_lengths)
                rle_values.append(row_rle_values)

        return rle_lengths, rle_values

    @staticmethod
    def decode(starts, lengths, values):
        """
        Decode a Run-Length Encoded (RLE) array.

        Parameters:
        starts (list): Start positions of RLE segments.
        lengths (list): Lengths of RLE segments.
        values (list): Values of RLE segments.

        Returns:
        numpy array: Decoded array.
        """
        starts, lengths, values = map(np.asarray, (starts, lengths, values))
        ends = starts + lengths
        n = ends[-1]

        decoded_array = np.full(n, np.nan)
        for lo, hi, val in zip(starts, ends, values):
            decoded_array[lo:hi] = val
        return decoded_array

    @staticmethod
    def hv_decode(rle_lengths, rle_values, output_shape, axis=1):
        """
        Decode a Run-Length Encoded (RLE) image along specified axis.

        Parameters:
        rle_lengths (list): RLE lengths.
        rle_values (list): RLE values.
        output_shape (tuple): Shape of the output image.
        axis (int): Axis along which to perform RLE decoding (0 for rows, 1 for columns).

        Returns:
        numpy array: Decoded image.
        """
        starts = [[int(np.sum(arr[:i])) for i in range(len(arr))] for arr in rle_lengths]

        decoded_image = np.zeros(output_shape, dtype=np.int32)
        if axis == 1:
            for i in range(decoded_image.shape[1]):
                decoded_image[:, i] = RLE.decode(starts[i], rle_lengths[i], rle_values[i])
        else:
            for i in range(decoded_image.shape[0]):
                decoded_image[i] = RLE.decode(starts[i], rle_lengths[i], rle_values[i])

        return decoded_image

    @staticmethod
    def calculate_pair_sum(array):
        """
        Calculate the sum of pairs in an array.

        Parameters:
        array (list): Input array.

        Returns:
        list: Array of pair sums.
        """
        if len(array) == 1:
            return list(array)
        else:
            pair_sums = [array[i] + array[i + 1] for i in range(0, len(array) - 1, 2)]
            if len(array) % 2 == 1:
                pair_sums.append(array[-2] + array[-1])
            return pair_sums

    @staticmethod
    def get_most_common(rle_lengths):
        """
        Get the most common pair sum in RLE.

        Parameters:
        rle_lengths (list): RLE lengths.

        Returns:
        int: Most common pair sum.
        """
        pair_sums = [RLE.calculate_pair_sum(col) for col in rle_lengths]

        flattened = []
        for col in pair_sums:
            flattened += col

        most_common = np.argmax(np.bincount(flattened))
        return most_common

    @staticmethod
    def most_common_bw_pattern(array, most_common):
        """
        Get the most common black-white pattern in RLE.

        Parameters:
        array (list): RLE lengths.
        most_common (int): Most common pair sum.

        Returns:
        list: Most common black-white patterns.
        """
        if len(array) == 1:
            return []
        else:
            patterns = [(array[i], array[i + 1]) for i in range(0, len(array) - 1, 2)
                        if array[i] + array[i + 1] == most_common]

            if len(array) % 2 == 1 and array[-2] + array[-1] == most_common:
                patterns.append((array[-2], array[-1]))
            return patterns