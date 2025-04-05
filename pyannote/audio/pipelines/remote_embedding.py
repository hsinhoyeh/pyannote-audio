# Modified portion of SpeakerDiarization to use remote embedding

# pyannote/audio/pipelines/remote_embedding.py
import torch
import pickle
import logging
import math
import time
import queue
import threading
import numpy as np
from typing import Iterator, Dict, Any, Optional, Callable, List, Tuple
from einops import rearrange

from pyannote.core import SlidingWindowFeature
from cockburn.prefetchd.client import PrefetchClient

logger = logging.getLogger(__name__)

def remote_embedding_generator(embedding_model):
    """Generator function for remote embedding computation.
    
    This generator function processes a stream of (waveform, mask) pairs
    and returns the computed embeddings.
    
    Args:
        embedding_model: The embedding model to use
        
    Returns:
        A generator that processes waveform and mask batches
    """
    logger.info(f"Initialized remote_embedding_generator")
    
    batch_index = 0
    
    # This is an infinite loop that processes each batch as it's received
    while True:
        # Get the next batch from the client through the generator protocol
        serialized_batch = yield
        
        if serialized_batch is None:
            logger.info("Received None, ending stream")
            break
        
        start_time = time.time()
        
        try:
            # Deserialize the input
            waveform_batch, mask_batch = pickle.loads(serialized_batch)
            
            # Perform the embedding computation (the heavy computation)
            embedding_batch = embedding_model(waveform_batch, masks=mask_batch)
            
            # Serialize the result with sequence information
            result = pickle.dumps({
                "batch_index": batch_index,
                "embedding": embedding_batch
            })
            
            processing_time = time.time() - start_time
            logger.debug(f"Processed batch {batch_index} in {processing_time:.2f}s")
            
            # Yield the serialized result
            yield result
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_index}: {e}")
            # Yield an error marker with sequence information
            error_info = {
                "error": str(e), 
                "batch_index": batch_index
            }
            yield pickle.dumps(error_info)
        
        batch_index += 1

def get_embeddings_remote(
    self,
    file,
    binary_segmentations: SlidingWindowFeature,
    exclude_overlap: bool = False,
    hook: Optional[Callable] = None,
):
    """Extract embeddings using a streaming approach with remote execution
    
    Parameters
    ----------
    file : AudioFile
    binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        Binarized segmentation.
    exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
    hook: Optional[Callable]
        Called during embeddings after every batch to report the progress
        
    Returns
    -------
    embeddings : (num_chunks, num_speakers, dimension) array
    """
    
    # Cache check (same as original implementation)
    if self.training:
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
    
    # Initialize client for remote execution
    client = PrefetchClient(
        port=50051,  # Use the default port or configure as needed
        auto_start_server=True
    )
    
    # Initialize with remote code execution
    # Using the function instead of a class instance - this has a __name__ attribute
    generator_func = remote_embedding_generator
    
    success = client.initialize_with_code(
        generator_func,
        params={"embedding_model": self._embedding},  # Pass the embedding model as a parameter
        timeout=120  # Longer timeout for initialization
    )
    
    if not success:
        logger.error("Failed to initialize remote embedding generator")
        client.close(stop_server=True)
        raise RuntimeError("Failed to initialize remote embedding generator")
    
    logger.info("Remote embedding generator initialized successfully")
    
    # Set up result collection
    embedding_dim = self._embedding.dimension
    embedding_batches = []  # Will contain all embedding batches in order
    
    # Process data in streaming fashion
    # We'll track which chunks and speakers each batch corresponds to
    batch_mapping = []  # Will contain (chunk_idx, speaker_idxs) for each batch
    
    try:
        # Process segments
        current_batch_waveforms = []
        current_batch_masks = []
        current_batch_mapping = []  # Track which (chunk,speaker) each item belongs to
        
        batch_count = 0
        total_items = 0
        
        if hook is not None:
            # We don't know the total yet, so we'll update it later
            hook("embeddings", None, total=1, completed=0)
        
        # First pass: prepare batches and send them
        for chunk_idx, ((chunk, masks), (_, clean_masks)) in enumerate(zip(binary_segmentations, clean_segmentations)):
            waveform, _ = self._audio.crop(
                file,
                chunk,
                duration=duration,
                mode="pad",
            )
            
            # mask may contain NaN (in case of partial stitching)
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
            
            for speaker_idx, (mask, clean_mask) in enumerate(zip(masks.T, clean_masks.T)):
                if np.sum(clean_mask) > min_num_frames:
                    used_mask = clean_mask
                else:
                    used_mask = mask
                
                current_batch_waveforms.append(waveform[None])  # (1, 1, num_samples)
                current_batch_masks.append(torch.from_numpy(used_mask)[None])  # (1, num_frames)
                current_batch_mapping.append((chunk_idx, speaker_idx))
                total_items += 1
                
                # Create batch when we reach batch_size
                if len(current_batch_waveforms) >= self.embedding_batch_size:
                    waveform_batch = torch.vstack(current_batch_waveforms)
                    mask_batch = torch.vstack(current_batch_masks)
                    
                    # Store the mapping for this batch
                    batch_mapping.append(current_batch_mapping)
                    
                    # Serialize the batch
                    serialized_batch = pickle.dumps((waveform_batch, mask_batch))
                    
                    # Send to server
                    # The server processes each batch and returns results
                    next_data, _ = client.get_next(timeout=10)  # Get any pending result
                    if next_data is not None:
                        # We have a result, deserialize it
                        result = pickle.loads(next_data)
                        if isinstance(result, dict) and "embedding" in result:
                            embedding_batches.append(result["embedding"])
                            
                            if hook is not None:
                                hook("embeddings", result["embedding"], 
                                     total=total_items, completed=len(embedding_batches))
                        elif isinstance(result, dict) and "error" in result:
                            logger.error(f"Error in batch {result.get('batch_index', 'unknown')}: {result['error']}")
                    
                    batch_count += 1
                    
                    # Reset for next batch
                    current_batch_waveforms = []
                    current_batch_masks = []
                    current_batch_mapping = []
        
        # Process any remaining items in the final batch
        if current_batch_waveforms:
            waveform_batch = torch.vstack(current_batch_waveforms)
            mask_batch = torch.vstack(current_batch_masks)
            
            # Store the mapping for this batch
            batch_mapping.append(current_batch_mapping)
            
            # Serialize the batch
            serialized_batch = pickle.dumps((waveform_batch, mask_batch))
            
            # Send to server
            next_data, _ = client.get_next(timeout=10)
            if next_data is not None:
                result = pickle.loads(next_data)
                if isinstance(result, dict) and "embedding" in result:
                    embedding_batches.append(result["embedding"])
                    
                    if hook is not None:
                        hook("embeddings", result["embedding"], 
                             total=total_items, completed=len(embedding_batches))
                elif isinstance(result, dict) and "error" in result:
                    logger.error(f"Error in batch {result.get('batch_index', 'unknown')}: {result['error']}")
            
            batch_count += 1
        
        # Collect all remaining results
        while len(embedding_batches) < batch_count:
            next_data, has_more = client.get_next(timeout=10)
            
            if next_data is not None:
                result = pickle.loads(next_data)
                if isinstance(result, dict) and "embedding" in result:
                    embedding_batches.append(result["embedding"])
                    
                    if hook is not None:
                        hook("embeddings", result["embedding"], 
                             total=total_items, completed=len(embedding_batches))
                elif isinstance(result, dict) and "error" in result:
                    logger.error(f"Error in batch {result.get('batch_index', 'unknown')}: {result['error']}")
            
            if not has_more:
                break
        
        # Check if we got all the expected results
        if len(embedding_batches) != batch_count:
            logger.warning(f"Expected {batch_count} batches but got {len(embedding_batches)}")
        
        # Now we need to reconstruct the full embeddings array from the batches
        # using the batch_mapping to place each embedding in the correct position
        result_embeddings = np.zeros((num_chunks, num_speakers, embedding_dim))
        
        for batch_idx, (batch, mapping) in enumerate(zip(embedding_batches, batch_mapping)):
            for item_idx, (chunk_idx, speaker_idx) in enumerate(mapping):
                if item_idx < len(batch):  # Safeguard against any size mismatches
                    result_embeddings[chunk_idx, speaker_idx] = batch[item_idx]
        
        # Cache the embeddings for subsequent trials
        if self.training:
            if self._segmentation.model.specifications.powerset:
                file["training_cache/embeddings"] = {
                    "embeddings": result_embeddings,
                }
            else:
                file["training_cache/embeddings"] = {
                    "segmentation.threshold": self.segmentation.threshold,
                    "embeddings": result_embeddings,
                }
        
        return result_embeddings
        
    except Exception as e:
        logger.error(f"Error in get_embeddings_remote: {e}")
        raise
    finally:
        # Always close the client
        client.close(stop_server=True)
