from typing import Tuple, Optional

import cv2


class ResizeOptions:
    """
    A class to represent various resize options for passing images to ANN

    Attributes:
    ----------
    chunking_mode: int
        Use CHUNKING_MODE_* constants of ResizeOptions to fill correctly.
        CHUNKING_MODE_STATIC sets static chunk size in pixels.
        CHUNKING_MODE_DYNAMIC sets chunk size in ratio of a picture full dimensions to chunk's size.
        It'll become more clear as you read along.

    chunk_size: Optional[Tuple[int, int]]
        For static chunking mode this sets actual size of a chunk in pixels.
            * So 72x72 image with chunk_size (24, 24) will be cut into 9 pieces 24x24 each.
        For dynamic chunking mode this sets ratio of source image dimensions to chunk's size.
            * With chunk_size (2, 2) a 72x72 picture will be divided into chunks of 36x36 (2 chunks horizontally, 2 vertically).
            * Whit chunk_size (3, 1) a 72x72 picture will be cut into chunks of 24x72 (3 chunks horizontally, 1 vertically).
            * You can not omit chunk_size for dynamic chunking mode, it will end in error.

    model_input_size: Tuple[int, int]
        The final size of a chunk that will be enforced before sending it to ANN. If omitted will be taken from model
        metadata eventually.

    resize_interpolation: int
        Use cv2.INTER_* constants to fill correctly. Default value: cv2.INTER_NEAREST
        Represents the interpolation algorythm that will be used in forced resize to model_input_size
    """

    CHUNKING_MODE_STATIC = 0
    CHUNKING_MODE_DYNAMIC = 1

    __allowed_chunking_modes = [CHUNKING_MODE_STATIC, CHUNKING_MODE_DYNAMIC]

    def __init__(self, chunking_mode: int = 0,
                 chunk_size: Optional[Tuple[int, int]] = None,
                 model_input_size: Optional[Tuple[int, int]] = None,
                 chunk2net_resize_interpolation: int = cv2.INTER_AREA,
                 net2map_resize_interpolation: int = cv2.INTER_NEAREST):
        if chunking_mode not in self.__allowed_chunking_modes:
            raise ValueError('Unknown chunking mode')

        if chunking_mode == self.CHUNKING_MODE_DYNAMIC and (chunk_size is None or 0 in chunk_size):
            raise ValueError('Chunking mode CHUNKING_MODE_DYNAMIC can not be used with undefined or zero chunk size')

        self.chunking_mode = chunking_mode
        self.chunk_size: Optional[Tuple[int, int]] = chunk_size
        self.model_input_size: Optional[Tuple[int, int]] = model_input_size
        self.chunk2net_resize_interpolation = chunk2net_resize_interpolation
        self.net2map_resize_interpolation = net2map_resize_interpolation

    def get_dynamic_chunk_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        if self.chunking_mode != self.CHUNKING_MODE_DYNAMIC:
            raise ValueError(f'''Can't  calculate dynamic chunk size if chunking mode not CHUNKING_MODE_DYNAMIC''')

        if original_size[0] % self.chunk_size[0] != 0 or original_size[1] % self.chunk_size[0] != 0:
            raise ValueError(f'''Can't calculate dynamic chunk size: original size can't be divided 
            entirely by chunk size''')

        return original_size[0] // self.chunk_size[0], original_size[1] // self.chunk_size[1]

    @staticmethod
    def default() -> "ResizeOptions":
        opts = ResizeOptions(chunking_mode=ResizeOptions.CHUNKING_MODE_STATIC,
                             chunk_size=(256, 256),
                             model_input_size=None)
        return opts
