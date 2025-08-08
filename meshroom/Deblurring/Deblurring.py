__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

# models paths
FMANET_MODEL_PATH = os.getenv('FMANET_MODEL_PATH')

class DeblurringNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        extension = extension_param.value
        input_path = input_path_param.value
        image_paths = get_image_paths_list(input_path, extension)

        return(max(1, len(image_paths)))


class DepthEstimationBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        # when block size is 0, only one block is used
        else:
            return size, size, 1

class Deblurring(desc.Node):
    category = "Deblurring"
    documentation = """This node deblurs an image sequence.
                    The FMA-Net model deblurs each image using the previous and the next images in the sequence."""
    
    gpu = desc.Level.INTENSIVE

    size = DeblurringNodeSize(['imagesFolder', 'inputExtension'])
    parallelization = DepthEstimationBlockSize()

    inputs = [
        desc.File(
            name="imagesFolder",
            label="Images Folder",
            description="Input images to deblur.",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input. If \"\" is selected, the provided input will be used as such.",
            values=["jpg", "jpeg", "png", "exr"],
            value="exr",
            exclusive=True,
        ),
        desc.IntParam(
            name="maxImageSize",
            label="Max Image Size",
            value=640,
            description="Sets the maximal image size. If input images have at least one dimension higher than this value, \n"
                        "they will be resized in order their higher dimension (width or height) matches it",
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the deblurred images stored with the same extension as input images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="deblurred",
            label="Deblurred images",
            description="Images with reduced motion blur",
            semantic="imageList",
            value=lambda attr: "{nodeCacheFolder}/*."+attr.node.inputExtension.value,
            group="",
        ),
    ]

    def processChunk(self, chunk):

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.imagesFolder.value:
                chunk.logger.warning('No input folder given.')

            # inference

            chunk.logger.info(f'Starting inference on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            from FMANet_utils.FMANet_inference import FMANet_inference

            FMANet_inference(
                inPath=chunk.node.imagesFolder.value,
                ext=chunk.node.inputExtension.value,
                checkpoint_path=FMANET_MODEL_PATH,
                outPath=chunk.node.output.value,
                maxDim=chunk.node.maxImageSize.value,
                start=chunk.range.start,
                stop=chunk.range.end)
            
            chunk.logger.info('Publish end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pathlib import Path
    import itertools

    include_suffices = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        raise ValueError(f"Input path '{input_path}' is not a directory.")
    return image_paths
