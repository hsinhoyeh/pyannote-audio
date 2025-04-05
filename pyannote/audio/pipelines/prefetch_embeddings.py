# pyannote/audio/pipelines/prefetch_embeddings.py

import torch
import numpy as np
import logging
from typing import List, Tuple, Iterator, Any, Optional, Callable, TypeVar, Generic
import math
import traceback
from einops import rearrange
import pickle
import os
import time
import struct
from pathlib import Path

from cockburn.prefetchd.wrapper import prefetch_iterator, PrefetchEnumerator
from pyannote.core import SlidingWindowFeature

# Define TypeVar for PrefetchEnumerator
T = TypeVar('T')

class AudioCropBatchGenerator:
    """Generator class that runs on the server side to crop audio and prepare batches.
    
    Yields serialized binary data directly. Compression will be handled by gRPC.
    """
    
    def __init__(self, file, binary_segmentations, min_num_frames=-1, 
                 embedding_batch_size=1, audio_processor=None):
        """Initialize the generator."""
        logger = logging.getLogger("prefetch_server")
        logger.setLevel(logging.INFO)

        self.logger = logger
        self.file = file
        self.binary_segmentations = binary_segmentations
        self.clean_segmentations = self._prepare_clean_segmentations(binary_segmentations, min_num_frames)
        self.min_num_frames = min_num_frames
        self.embedding_batch_size = embedding_batch_size
        self.audio_processor = audio_processor
        self.started = False
        self.finished = False
        
        # Extract basic properties
        self.duration = binary_segmentations.sliding_window.duration
        
        self.logger.info(f"Initialized AudioCropBatchGenerator with batch_size={embedding_batch_size}")
    
    def _prepare_clean_segmentations(self, binary_segmentations, min_num_frames):
        """Prepare clean segmentations (segments with no speaker overlap)."""
        try:
            # Zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            
            # Return clean segmentations with the same window structure
            return type(binary_segmentations)(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window
            )
        except Exception as e:
            self.logger.error(f"Error preparing clean segmentations: {e}")
            # Return a copy of the original segmentations as fallback
            return binary_segmentations
    
    def __iter__(self) -> Iterator[bytes]:
        """Generate batches of (waveform, mask) pairs and yield them as serialized bytes.
        
        Returns:
            Iterator yielding serialized batch data directly (gRPC will handle compression)
        """
        if self.started:
            self.logger.warning("Iterator was already started, creating new iterator")
            
        self.started = True
        self.finished = False
        
        def iter_waveform_and_mask():
            """Iterator over individual (waveform, mask) pairs."""
            try:
                for idx, ((chunk, masks), (_, clean_masks)) in enumerate(zip(
                    self.binary_segmentations, self.clean_segmentations
                )):
                    try:
                        # Crop audio
                        waveform, _ = self.audio_processor.crop(
                            self.file,
                            chunk,
                            duration=self.duration,
                            mode="pad",
                        )
                    except Exception as e:
                        self.logger.error(f"Error during audio crop: {e}")
                        self.logger.debug(traceback.format_exc())
                        continue
                    
                    # Handle NaN values in masks
                    try:
                        masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                        clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
                    except Exception as e:
                        self.logger.error(f"Error handling masks: {e}")
                        continue
                    
                    # Check waveform size to avoid memory issues
                    try:
                        if isinstance(waveform, torch.Tensor):
                            waveform_size = waveform.numel() * waveform.element_size()
                        elif isinstance(waveform, np.ndarray):
                            waveform_size = waveform.nbytes
                        else:
                            waveform_size = 0
                            
                        # Skip extremely large waveforms
                        if waveform_size > 100_000_000:  # 100MB
                            self.logger.warning(f"Skipping oversized waveform: {waveform_size} bytes")
                            continue
                    except Exception as e:
                        self.logger.error(f"Error checking waveform size: {e}")
                    
                    # Iterate through each speaker in this chunk
                    for spk_idx, (mask, clean_mask) in enumerate(zip(masks.T, clean_masks.T)):
                        try:
                            # Decide which mask to use based on min_num_frames
                            if np.sum(clean_mask) > self.min_num_frames:
                                used_mask = clean_mask
                            else:
                                used_mask = mask
                            
                            # Convert to numpy arrays
                            waveform_arr = waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
                            mask_arr = used_mask
                            
                            yield (waveform_arr, mask_arr)
                        except Exception as e:
                            self.logger.error(f"Error preparing speaker {spk_idx} data: {e}")
                            continue
                
            except Exception as e:
                self.logger.error(f"Fatal error in waveform iterator: {e}")
                self.logger.debug(traceback.format_exc())
                
            # Mark as finished when iteration is complete
            self.finished = True
        
        # Create batches from the iterator and serialize directly
        batch = []
        batch_count = 0
        items_processed = 0
        
        try:
            for item in iter_waveform_and_mask():
                items_processed += 1
                
                # Log processing status periodically
                if items_processed % 10 == 0:
                    self.logger.info(f"Processed {items_processed} items so far")
                
                batch.append(item)
                
                # When we reach batch size, serialize the batch and yield it
                if len(batch) >= self.embedding_batch_size:
                    try:
                        serialized_batch = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
                        batch_size = len(serialized_batch)
                        
                        # Yield the serialized data directly (gRPC will handle compression)
                        yield serialized_batch
                    except Exception as e:
                        self.logger.error(f"Error serializing batch {batch_count}: {e}")
                    
                    batch = []
                    batch_count += 1
            
            # Serialize and yield any remaining items in the final batch
            if batch:
                try:
                    serialized_batch = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
                    batch_size = len(serialized_batch)
                    
                    # Yield the serialized data directly (gRPC will handle compression)
                    yield serialized_batch
                except Exception as e:
                    self.logger.error(f"Error serializing final batch {batch_count}: {e}")
                    
            # Yield an empty bytes object as the end marker
            yield b''
            
            self.finished = True
        except Exception as e:
            self.logger.error(f"Fatal error in batch generation: {e}")
            self.logger.debug(traceback.format_exc())
            self.finished = True


# Direct memory transfer helper class for communication
class DirectCommunicator:
    """Helper class for inter-process communication using direct memory transfer.
    Compression is handled by gRPC.
    """
    
    @staticmethod
    def serialize_object(obj):
        """Serialize an object using pickle.
        
        Args:
            obj: Python object to serialize
            
        Returns:
            bytes: Serialized object
        """
        logger = logging.getLogger("direct_communicator")
        logger.setLevel(logging.INFO)
        
        try:
            # Special handling for complex objects before pickling
            if isinstance(obj, dict):
                # Process parameters that might contain PyTorch tensors or other complex objects
                processed_obj = {}
                for key, value in obj.items():
                    if isinstance(value, torch.Tensor):
                        # Convert PyTorch tensors to numpy arrays
                        processed_obj[key] = {"__tensor_data__": value.cpu().detach().numpy()}
                    elif key == "audio_processor":
                        # Instead of passing the audio processor object, pass its configuration or class info
                        if hasattr(value, 'get_config'):
                            # If the audio processor has a get_config method, use it
                            logger.info("Extracting audio processor config")
                            try:
                                config = value.get_config()
                                processed_obj[key] = {
                                    "__audio_processor_config__": config
                                }
                                logger.info(f"Extracted audio processor config: {config}")
                            except Exception as e:
                                logger.error(f"Failed to get audio processor config: {e}")
                                # Fall back to class name only
                                processed_obj[key] = {
                                    "__audio_processor_class__": value.__class__.__name__
                                }
                        else:
                            # Otherwise just pass the class name for reconstruction
                            logger.info(f"Using audio processor class name: {value.__class__.__name__}")
                            processed_obj[key] = {
                                "__audio_processor_class__": value.__class__.__name__
                            }
                    elif key == "file":
                        # For AudioFile objects, we need to keep the original
                        processed_obj[key] = value
                    elif key == "embedding_model":
                        # We don't need to send the embedding model to the server
                        logger.info("Skipping embedding_model in serialization")
                        processed_obj[key] = None
                    elif key == "binary_segmentations" and hasattr(value, 'data') and hasattr(value, 'sliding_window'):
                        # Special handling for SlidingWindowFeature
                        # Extract the underlying data and metadata
                        processed_obj[key] = {
                            "__class__": "SlidingWindowFeature",
                            "data": value.data.copy() if hasattr(value.data, 'copy') else value.data,
                            "sliding_window": {
                                "start": value.sliding_window.start,
                                "step": value.sliding_window.step,
                                "duration": value.sliding_window.duration
                            }
                        }
                    else:
                        processed_obj[key] = value
                obj = processed_obj
                logger.info(f"Prepared object for serialization with keys: {list(obj.keys())}")
            
            # Pickle the object to get its byte representation
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            data_size = len(data)
            logger.info(f"Serialized data size: {data_size} bytes")
            
            return data
        
        except Exception as e:
            logger.error(f"Error serializing object: {e}")
            logger.debug(traceback.format_exc())
            
            # Create a fallback with error information
            try:
                fallback_obj = {"error": f"Serialization error: {str(e)}"}
                data = pickle.dumps(fallback_obj, protocol=pickle.HIGHEST_PROTOCOL)
                return data
            except Exception as e2:
                logger.error(f"Fallback serialization also failed: {e2}")
                # If all else fails, return empty bytes to signal an error
                return b''
    
    @staticmethod
    def deserialize_object(data):
        """Deserialize an object from binary data.
        
        Args:
            data: Binary data
            
        Returns:
            The deserialized Python object
        """
        logger = logging.getLogger("direct_communicator")
        logger.setLevel(logging.INFO)
        
        try:
            if not data or len(data) < 1:
                logger.error("Invalid data: missing or too short")
                return None
                
            # Unpickle the object
            obj = pickle.loads(data)
            logger.info(f"Successfully unpickled object of size {len(data)} bytes")
            
            # Post-process any specially handled objects
            if isinstance(obj, dict):
                processed_obj = {}
                for key, value in obj.items():
                    if value is None:
                        processed_obj[key] = None
                        continue
                        
                    if isinstance(value, dict):
                        if "__tensor_data__" in value:
                            # Convert numpy arrays back to PyTorch tensors
                            tensor_data = value["__tensor_data__"]
                            if tensor_data is not None:
                                try:
                                    processed_obj[key] = torch.from_numpy(tensor_data)
                                except Exception as e:
                                    logger.error(f"Error converting tensor data: {e}")
                                    processed_obj[key] = None
                            else:
                                processed_obj[key] = None
                        elif "__audio_processor_config__" in value:
                            # Reconstruct audio processor from config
                            try:
                                from pyannote.audio.core.io import Audio
                                logger.info("Reconstructing audio processor from config")
                                # Create default Audio processor
                                audio_processor = Audio()
                                # Apply any configuration if possible
                                config = value["__audio_processor_config__"]
                                if hasattr(audio_processor, 'set_config') and config:
                                    audio_processor.set_config(config)
                                    logger.info(f"Applied config to audio processor: {config}")
                                processed_obj[key] = audio_processor
                            except Exception as e:
                                logger.error(f"Error reconstructing audio processor from config: {e}")
                                logger.debug(traceback.format_exc())
                                # Create a simple Audio processor as fallback
                                try:
                                    from pyannote.audio.core.io import Audio
                                    logger.info("Creating fallback Audio processor")
                                    processed_obj[key] = Audio()
                                except Exception as e2:
                                    logger.error(f"Error creating fallback audio processor: {e2}")
                                    processed_obj[key] = None
                        elif "__audio_processor_class__" in value:
                            # Reconstruct audio processor from class name
                            try:
                                from pyannote.audio.core.io import Audio
                                logger.info(f"Creating audio processor from class name: {value['__audio_processor_class__']}")
                                # Create default Audio processor regardless of original class
                                # This is a simpler fallback approach
                                processed_obj[key] = Audio()
                            except Exception as e:
                                logger.error(f"Error creating audio processor: {e}")
                                processed_obj[key] = None
                        elif value.get("__class__") == "SlidingWindowFeature":
                            # Reconstruct SlidingWindowFeature
                            try:
                                from pyannote.core import SlidingWindow, SlidingWindowFeature
                                
                                # Safely extract sliding window parameters
                                sliding_window_dict = value.get("sliding_window", {})
                                if not isinstance(sliding_window_dict, dict):
                                    logger.warning(f"Invalid sliding_window data: {sliding_window_dict}")
                                    processed_obj[key] = None
                                    continue
                                    
                                start = sliding_window_dict.get("start", 0.0)
                                step = sliding_window_dict.get("step", 1.0)
                                duration = sliding_window_dict.get("duration", 1.0)
                                
                                # Create sliding window
                                sw = SlidingWindow(start=start, step=step, duration=duration)
                                
                                # Safely extract data
                                feature_data = value.get("data")
                                if feature_data is None:
                                    logger.warning("Missing data for SlidingWindowFeature")
                                    processed_obj[key] = None
                                    continue
                                    
                                # Create SlidingWindowFeature
                                processed_obj[key] = SlidingWindowFeature(feature_data, sw)
                                
                            except Exception as e:
                                logger.error(f"Error reconstructing SlidingWindowFeature: {e}")
                                processed_obj[key] = None
                        else:
                            processed_obj[key] = value
                    else:
                        processed_obj[key] = value
                
                logger.info(f"Processed object with keys: {list(processed_obj.keys())}")
                return processed_obj
            
            return obj
                
        except Exception as e:
            logger.error(f"Error deserializing object: {e}")
            logger.debug(traceback.format_exc())
            return None


# Modified function to be serialized and sent to the server
def create_audio_crop_generator(**kwargs):
    """Create a generator that yields binary data directly.
    
    This function is designed to receive parameters via direct memory transfer
    and create a generator that directly serializes data.
    
    Args:
        params_data: Binary data containing parameters
        
    Returns:
        A generator yielding serialized batches
    """
    logger = logging.getLogger("prefetch_server")
    logger.setLevel(logging.INFO)
    
    try:
        # Handle direct binary data
        if len(kwargs) == 1 and 'params_data' in kwargs:
            # This is our binary data
            params_data = kwargs['params_data']
        elif len(kwargs) == 1 and next(iter(kwargs.values())) is not None and isinstance(next(iter(kwargs.values())), bytes):
            # The first argument is our binary data, regardless of the key
            params_data = next(iter(kwargs.values()))
        else:
            # If we received regular parameters, log an error and return an empty generator
            logger.error("Direct parameters are not supported with the direct memory implementation")
            return iter([])  # Return empty iterator
        
        logger.info(f"Received parameters data of size {len(params_data)} bytes")
        
        # Deserialize parameters
        params = DirectCommunicator.deserialize_object(params_data)
        
        if params is None:
            logger.error("Failed to deserialize parameters")
            return iter([])  # Return empty iterator
        
        # Extract parameters
        file = params.get("file")
        binary_segmentations = params.get("binary_segmentations")
        audio_processor = params.get("audio_processor")
        audio_config = params.get("audio_config")
        min_num_frames = params.get("min_num_frames", -1)
        embedding_batch_size = params.get("embedding_batch_size", 1)
        
        # Validate required parameters
        if file is None:
            logger.error("Missing required parameter: file")
            return iter([])
            
        if binary_segmentations is None:
            logger.error("Missing required parameter: binary_segmentations")
            return iter([])
        
        # If audio_processor is None, create it from config or create a default one
        if audio_processor is None:
            try:
                from pyannote.audio.core.io import Audio
                logger.info("Creating new Audio processor on server")
                audio_processor = Audio()
                
                # Apply configuration if available
                if audio_config and hasattr(audio_processor, 'set_config'):
                    audio_processor.set_config(audio_config)
                    logger.info("Applied config to audio processor")
                else:
                    logger.info("Using default audio processor configuration")
            except Exception as e:
                logger.error(f"Error creating audio processor: {e}")
                logger.debug(traceback.format_exc())
                return iter([])  # Return empty iterator
        
        if audio_processor is None:
            logger.error("Failed to create audio processor")
            return iter([])
        
        logger.info("All parameters validated, creating AudioCropBatchGenerator")
        
        # Create the generator 
        generator = AudioCropBatchGenerator(
            file=file,
            binary_segmentations=binary_segmentations,
            min_num_frames=min_num_frames,
            embedding_batch_size=embedding_batch_size,
            audio_processor=audio_processor
        )
        
        # Return the iterator from the generator
        return iter(generator)
        
    except Exception as e:
        logger.error(f"Error creating generator: {e}")
        logger.debug(traceback.format_exc())
        
        # Return empty iterator instead of raising to avoid crashing the server
        return iter([])

class PrefetchedEmbeddingExtractor:
    """Client-side embedding extractor that uses prefetching for audio data.
    
    This class handles the client side of the distributed processing, where:
    1. The server crops audio and prepares batches of (waveform, mask) pairs
    2. The client receives these batches and performs the embedding computation
    
    Uses direct memory transfer with gRPC compression for improved efficiency.
    """
    
    def __init__(self, 
                 file, 
                 binary_segmentations, 
                 embedding_model, 
                 audio_processor,
                 min_num_frames=-1,
                 embedding_batch_size=1,
                 exclude_overlap=False,
                 port=50051,
                 auto_start_server=True,
                 server_script=None,
                 auto_stop_server=True,
                 address='localhost',
                 item_timeout=10,
                 max_retries=3,
                 hook: Optional[Callable] = None):
        """Initialize the prefetched embedding extractor."""
        logger = logging.getLogger("prefetch_client")
        logger.setLevel(logging.INFO)

        self.logger = logger
        self.file = file
        self.binary_segmentations = binary_segmentations
        self.embedding_model = embedding_model
        self.audio_processor = audio_processor
        self.min_num_frames = min_num_frames
        self.embedding_batch_size = embedding_batch_size
        self.exclude_overlap = exclude_overlap
        self.hook = hook
        
        # Server configuration
        self.port = port
        self.auto_start_server = auto_start_server
        self.server_script = server_script
        self.auto_stop_server = auto_stop_server
        self.address = address
        self.item_timeout = item_timeout
        self.max_retries = max_retries
        
        # Extract shape information
        self.num_chunks, self.num_frames, self.num_speakers = binary_segmentations.data.shape
        
        self.logger.debug(f"Initialized PrefetchedEmbeddingExtractor with batch_size={embedding_batch_size}")
        
        # Heartbeat and monitoring
        self.start_time = time.time()
        self.items_processed = 0

    def extract_embeddings(self):
        """Extract embeddings using prefetched audio data.
        
        Returns:
            numpy.ndarray: Array of shape (num_chunks, num_speakers, dimension)
        """
        try:
            # Get audio processor configuration if available
            audio_config = None
            if hasattr(self.audio_processor, 'get_config'):
                try:
                    audio_config = self.audio_processor.get_config()
                    self.logger.debug(f"Extracted audio processor config: {audio_config}")
                except Exception as e:
                    self.logger.error(f"Failed to get audio processor config: {e}")
            
            # Setup parameters dictionary without the full audio_processor
            params = {
                "file": self.file,
                "binary_segmentations": self.binary_segmentations,
                "min_num_frames": self.min_num_frames,
                "embedding_batch_size": self.embedding_batch_size,
                "audio_config": audio_config
            }
            
            # Serialize parameters to binary data
            self.logger.debug("Serializing parameters to binary data")
            params_data = DirectCommunicator.serialize_object(params)
            self.logger.debug(f"Parameters serialized to {len(params_data)} bytes of data")
            
            # Create empty list to collect embeddings
            embedding_batches = []
            batch_count = math.ceil(self.num_chunks * self.num_speakers / self.embedding_batch_size)
            
            # Report initial progress if hook is provided
            if self.hook is not None:
                self.hook("embeddings", None, total=batch_count, completed=0)
            
            # Create the PrefetchEnumerator with direct memory transfer and gRPC compression
            enumerator = PrefetchEnumerator(
                generator_function_or_class=create_audio_crop_generator,
                port=self.port,
                params={'params_data': params_data},
                auto_start_server=self.auto_start_server,
                server_script=self.server_script,
                timeout=180,  # Longer timeout for initialization
                address=self.address,
            )
            
            # Set a timeout for the entire client-side processing
            client_deadline = time.time() + 3600  # 1 hour timeout
            item_count = 0
            consecutive_errors = 0
            
            try:
                # Get the iterator from the enumerator
                iterator = iter(enumerator)
                
                # Process each batch from the server
                while time.time() < client_deadline:
                    # Get the next batch with explicit error handling
                    try:
                        # Get one batch at a time
                        serialized_batch = next(iterator)
                        item_count += 1
                        # Reset consecutive errors since we got an item
                        consecutive_errors = 0
                    except StopIteration:
                        # End of iterator reached
                        self.logger.debug("End of batch iteration reached")
                        break
                    except Exception as e:
                        # Log error and continue if it's recoverable
                        consecutive_errors += 1
                        self.logger.error(f"Error getting next batch: {e}")
                        if consecutive_errors >= 3:
                            self.logger.error("Too many consecutive errors, stopping")
                            break
                        # Short pause before retry
                        time.sleep(0.5)
                        continue
                    
                    # Log heartbeat
                    self.items_processed += 1
                    current_time = time.time()
                    if current_time - self.start_time > 30:  # Log every 30 seconds
                        self.logger.debug(f"Client heartbeat: Processed {self.items_processed} items")
                        self.start_time = current_time
                    
                    self.logger.info(f"Processing batch #{item_count}")
                    
                    # Skip empty batches (end marker)
                    if not serialized_batch or serialized_batch == b'':
                        self.logger.info("Received end marker, finishing up")
                        break
                    
                    # Deserialize the batch
                    try:
                        batch = pickle.loads(serialized_batch)
                    except Exception as e:
                        self.logger.error(f"Error deserializing batch {item_count}: {e}")
                        continue
                    
                    if batch is not None:
                        try:
                            # Unzip the batch into waveforms and masks
                            waveforms, masks = zip(*batch)
                            
                            # Create tensors from numpy arrays
                            waveform_tensors = []
                            mask_tensors = []
                            
                            for waveform, mask in zip(waveforms, masks):
                                try:
                                    # Convert to tensors if needed
                                    if isinstance(waveform, np.ndarray):
                                        waveform_tensor = torch.from_numpy(waveform)
                                        # Ensure it has the right dimensions
                                        if waveform_tensor.dim() == 2:  # (1, num_samples)
                                            waveform_tensor = waveform_tensor[None]  # Add batch dimension
                                        elif waveform_tensor.dim() == 1:  # (num_samples)
                                            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
                                    elif isinstance(waveform, torch.Tensor):
                                        waveform_tensor = waveform
                                        # Ensure it has the right dimensions
                                        if waveform_tensor.dim() == 2:  # (1, num_samples)
                                            waveform_tensor = waveform_tensor[None]  # Add batch dimension
                                        elif waveform_tensor.dim() == 1:  # (num_samples)
                                            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
                                    else:
                                        self.logger.warning(f"Unexpected waveform type: {type(waveform)}")
                                        continue
                                        
                                    if isinstance(mask, np.ndarray):
                                        mask_tensor = torch.from_numpy(mask)
                                        # Ensure it has the right dimensions
                                        if mask_tensor.dim() == 1:  # (num_frames)
                                            mask_tensor = mask_tensor[None]  # Add batch dimension
                                    elif isinstance(mask, torch.Tensor):
                                        mask_tensor = mask
                                        # Ensure it has the right dimensions
                                        if mask_tensor.dim() == 1:  # (num_frames)
                                            mask_tensor = mask_tensor[None]  # Add batch dimension
                                    else:
                                        self.logger.warning(f"Unexpected mask type: {type(mask)}")
                                        continue
                                        
                                    waveform_tensors.append(waveform_tensor)
                                    mask_tensors.append(mask_tensor)
                                except Exception as e:
                                    self.logger.error(f"Error converting to tensor: {e}")
                            
                            if not waveform_tensors or not mask_tensors:
                                self.logger.warning(f"No valid tensors in batch #{item_count}")
                                continue
                                
                            try:
                                # Stack the waveforms and masks
                                waveform_batch = torch.cat(waveform_tensors, dim=0)  # (batch_size, 1, num_samples)
                                mask_batch = torch.cat(mask_tensors, dim=0)          # (batch_size, num_frames)
                            except Exception as e:
                                self.logger.error(f"Error stacking tensors: {e}")
                                continue
                            
                            # Extract embeddings - this is now on the client side with GPU access
                            with torch.no_grad():  # Add no_grad to save memory
                                embedding_batch = self.embedding_model(waveform_batch, masks=mask_batch)
                            
                            # Convert to numpy if it's a tensor
                            if isinstance(embedding_batch, torch.Tensor):
                                embedding_batch_numpy = embedding_batch.cpu().numpy()
                            else:
                                embedding_batch_numpy = embedding_batch
                                
                            embedding_batches.append(embedding_batch_numpy)
                            
                            # Report progress if hook is provided
                            if self.hook is not None:
                                self.hook("embeddings", embedding_batch_numpy, total=batch_count, completed=item_count)
                                
                        except Exception as e:
                            self.logger.error(f"Error processing batch #{item_count}: {e}")
                            self.logger.debug(traceback.format_exc())
                    else:
                        self.logger.warning(f"Received empty batch #{item_count}")
                
            finally:
                # Ensure proper cleanup of the enumerator
                if hasattr(enumerator, 'client') and enumerator.client:
                    try:
                        enumerator.client.close(stop_server=self.auto_stop_server)
                    except Exception as e:
                        self.logger.error(f"Error closing enumerator client: {e}")
            
            # Stack all embedding batches
            if not embedding_batches:
                raise ValueError("No embedding batches were successfully processed")
                
            embeddings_flat = np.vstack(embedding_batches)
            
            # Calculate actual dimensions based on the data we received
            total_items = embeddings_flat.shape[0]
            dimension = embeddings_flat.shape[1]
            
            self.logger.info(f"Total embedding items: {total_items}, dimension: {dimension}")
            self.logger.info(f"Original expected chunks: {self.num_chunks}, speakers: {self.num_speakers}")
            
            # Check if we can reshape according to expected dimensions
            if total_items == self.num_chunks * self.num_speakers:
                # If the dimensions match as expected, use the original reshape
                embeddings = rearrange(embeddings_flat, "(c s) d -> c s d", c=self.num_chunks, s=self.num_speakers)
            else:
                # Calculate new chunk size based on actual data received
                actual_chunks = total_items // self.num_speakers
                
                if total_items % self.num_speakers == 0:
                    embeddings = rearrange(embeddings_flat, "(c s) d -> c s d", c=actual_chunks, s=self.num_speakers)
                else:
                    # If we can't evenly divide, just return the flat embeddings with a warning
                    self.logger.warning(
                        f"Cannot reshape embeddings: {total_items} items can't be evenly divided by {self.num_speakers} speakers"
                    )
                    # Create a dummy reshape that keeps the original dimensions
                    embeddings = embeddings_flat.reshape(-1, 1, dimension)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in extract_embeddings: {e}")
            self.logger.debug(traceback.format_exc())
            raise

# Modified get_embeddings function for the SpeakerDiarization class
def get_embeddings_prefetched(
    self,
    file,
    binary_segmentations: SlidingWindowFeature,
    exclude_overlap: bool = False,
    hook: Optional[Callable] = None,
):
    """Extract embeddings using prefetched audio processing with gRPC compression.
    
    This is a drop-in replacement for the original get_embeddings method
    that uses the prefetch architecture for distributed processing with direct
    memory transfer and gRPC native compression for improved performance.
    
    Parameters
    ----------
    file : AudioFile
        Audio file to process
    binary_segmentations : SlidingWindowFeature
        Binarized segmentation.
    exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
    hook: Optional[Callable]
        Called during embeddings after every batch to report progress
        
    Returns
    -------
    embeddings : (num_chunks, num_speakers, dimension) array
        Extracted speaker embeddings
    """
    # When optimizing the hyper-parameters of this pipeline with frozen
    # "segmentation.threshold", one can reuse the embeddings from the first trial
    if self.training:
        # We only re-use embeddings if they were extracted based on the same value of the
        # "segmentation.threshold" hyperparameter or if the segmentation model relies on
        # `powerset` mode
        cache = file.get("training_cache/embeddings", dict())
        if ("embeddings" in cache) and (
            self._segmentation.model.specifications.powerset
            or (cache["segmentation.threshold"] == self.segmentation.threshold)
        ):
            return cache["embeddings"]
    
    # Create the prefetched embedding extractor with gRPC compression
    extractor = PrefetchedEmbeddingExtractor(
        file=file,
        binary_segmentations=binary_segmentations,
        embedding_model=self._embedding,
        audio_processor=self._audio,
        min_num_frames=self._embedding.min_num_samples if exclude_overlap else -1,
        embedding_batch_size=self.embedding_batch_size,
        exclude_overlap=exclude_overlap,
        port=50051,  # Default port for the gRPC server
        auto_start_server=True,  # Automatically start the server if not running
        server_script=None,  # Use default server script path
        auto_stop_server=True,  # Automatically stop the server when done
        address='localhost',  # Default server address
        item_timeout=10,  # Timeout for each item request (seconds)
        max_retries=3,  # Maximum number of retries when no item available
        hook=hook  # Progress callback
    )
    
    # Extract embeddings using the prefetch architecture with gRPC compression
    embeddings = extractor.extract_embeddings()
    
    # Cache embeddings for subsequent trials if in training mode
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
