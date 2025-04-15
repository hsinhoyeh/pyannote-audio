# pyannote/audio/pipelines/prefetch_embeddings.py

import math
import numpy as np
import logging
import torch

from pyannote.audio.pipelines.speaker_diarization import batchify
from pyannote.core import SlidingWindowFeature
from einops import rearrange
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def get_embeddings(
    self,
    file,
    binary_segmentations: SlidingWindowFeature,
    exclude_overlap: bool = False,
    hook: Optional[Callable] = None,
):
    """Extract embeddings for each (chunk, speaker) pair

    Parameters
    ----------
    file : AudioFile
    binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        Binarized segmentation.
    exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        In case non-overlapping speech is too short, use the whole speech.
    hook: Optional[Callable]
        Called during embeddings after every batch to report the progress

    Returns
    -------
    embeddings : (num_chunks, num_speakers, dimension) array
    """

    # when optimizing the hyper-parameters of this pipeline with frozen
    # "segmentation.threshold", one can reuse the embeddings from the first trial,
    # bringing a massive speed up to the optimization process (and hence allowing to use
    # a larger search space).
    if self.training:
        # we only re-use embeddings if they were extracted based on the same value of the
        # "segmentation.threshold" hyperparameter or if the segmentation model relies on
        # `powerset` mode
        cache = file.get("training_cache/embeddings", dict())
        if ("embeddings" in cache) and (
            self._segmentation.model.specifications.powerset
            or (cache["segmentation.threshold"] == self.segmentation.threshold)
        ):
            return cache["embeddings"]

    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    if exclude_overlap:
        # minimum number of samples needed to extract an embedding
        # (a lower number of samples would result in an error)
        min_num_samples = self._embedding.min_num_samples

        # corresponding minimum number of frames
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

        # zero-out frames with overlapping speech
        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames,
            binary_segmentations.sliding_window,
        )

    else:
        min_num_frames = -1
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data, binary_segmentations.sliding_window
        )

    def iter_waveform_and_mask():
        for (chunk, masks), (_, clean_masks) in zip(
            binary_segmentations, clean_segmentations
        ):
            # chunk: Segment(t, t + duration)
            # masks: (num_frames, local_num_speakers) np.ndarray

            waveform, _ = self._audio.crop(
                file,
                chunk,
                duration=duration,
                mode="pad",
            )
            # waveform: (1, num_samples) torch.Tensor

            # mask may contain NaN (in case of partial stitching)
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

            for mask, clean_mask in zip(masks.T, clean_masks.T):
                # mask: (num_frames, ) np.ndarray

                if np.sum(clean_mask) > min_num_frames:
                    used_mask = clean_mask
                else:
                    used_mask = mask

                yield waveform[None], torch.from_numpy(used_mask)[None]
                # w: (1, 1, num_samples) torch.Tensor
                # m: (1, num_frames) torch.Tensor

    batches = batchify(
        iter_waveform_and_mask(),
        batch_size=self.embedding_batch_size,
        fillvalue=(None, None),
    )

    batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)
    embedding_batches = []

    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)
        
    # Define processor function for embedding computation
    def embedding_processor(data):
        waveform_batch, mask_batch = data
        return self._embedding(waveform_batch, masks=mask_batch)

    # Use JobManager for GPU-intensive task
    from cockburn.prefetchd.job_manager import JobManager
    from cockburn.prefetchd.prefetch_enumerator import PrefetchEnumerator
    from cockburn.prefetchd.prefetch_sender import PrefetchSender
    
    # Initialize JobManager with the embedding processor function
    with JobManager(
        processor_function=embedding_processor,
    ) as job_manager:
        
        # Use PrefetchSender to send jobs to the server
        with PrefetchSender(job_manager) as sender:
            for i, batch in enumerate(batches, 1):
                logger.info(f"Batch: {i} sending to processor")
                waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))
                
                if not waveforms:  # Skip empty batches
                    continue
                    
                waveform_batch = torch.vstack(waveforms)
                # (batch_size, 1, num_samples) torch.Tensor
                
                mask_batch = torch.vstack(masks)
                # (batch_size, num_frames) torch.Tensor
                
                # Send the job to the server via the sender
                # We need to package the data as a tuple to pass to our processor function
                sender.send((waveform_batch, mask_batch))
                logger.info(f"Batch: {i} sent to processor")
        
        # Use PrefetchEnumerator to collect results
        completed = 0
        with PrefetchEnumerator(job_manager) as enumerator:
            for i, embedding_batch in enumerate(enumerator, 1):
                embedding_batches.append(embedding_batch)
                
                if hook is not None:
                    completed += 1
                    hook("embeddings", embedding_batch, total=batch_count, completed=completed)
                logger.info(f"Batch: {i} processing completed")

    # Combine all batches
    embedding_batches = np.vstack(embedding_batches)
    embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

    # caching embeddings for subsequent trials
    # (see comments at the top of this method for more details)
    if self.training:
        if self._segmentation.model.specifications.powerset:
            file["training_cache/embeddings"] = {
                "embeddings": embeddings,
            }
        else:
            file["training_cache/embeddings"] = {
                "segmentation.threshold": self.segmentation.threshold,
                "embeddings": embeddings,
            }

    return embeddings
