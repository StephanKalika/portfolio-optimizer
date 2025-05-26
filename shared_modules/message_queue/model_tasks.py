"""
Model training task handlers for RabbitMQ messaging.

This module defines the structure of model training tasks
and provides helper functions for submitting and processing them.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .rabbitmq import (
    AsyncRabbitMQClient,
    MODEL_TRAINING_QUEUE,
    RESULT_NOTIFICATION_QUEUE,
    RabbitMQClient,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Enum for task status values."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskType(str, Enum):
    """Enum for model training task types."""
    
    TRAIN_MODEL = "train_model"
    PREDICT = "predict"
    EVALUATE = "evaluate"


class ModelTrainingTask:
    """Model training task for message queue."""
    
    def __init__(
        self,
        task_type: Union[TaskType, str],
        tickers: List[str],
        model_name: str,
        start_date: str,
        end_date: str,
        model_type: str = "lstm",
        sequence_length: int = 60,
        epochs: int = 50,
        hidden_layer_size: int = 50,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        test_split_size: float = 0.2,
        additional_params: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        created_at: Optional[str] = None,
        status: Union[TaskStatus, str] = TaskStatus.PENDING,
    ):
        """Initialize a model training task."""
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type if isinstance(task_type, str) else task_type.value
        self.status = status if isinstance(status, str) else status.value
        self.tickers = tickers
        self.model_name = model_name
        self.start_date = start_date
        self.end_date = end_date
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_split_size = test_split_size
        self.additional_params = additional_params or {}
        self.user_id = user_id
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = self.created_at
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "tickers": self.tickers,
            "model_name": self.model_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "epochs": self.epochs,
            "hidden_layer_size": self.hidden_layer_size,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "test_split_size": self.test_split_size,
            "additional_params": self.additional_params,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelTrainingTask":
        """Create a task from a dictionary."""
        return cls(
            task_id=data.get("task_id"),
            task_type=data.get("task_type", TaskType.TRAIN_MODEL),
            status=data.get("status", TaskStatus.PENDING),
            tickers=data.get("tickers", []),
            model_name=data.get("model_name", ""),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            model_type=data.get("model_type", "lstm"),
            sequence_length=data.get("sequence_length", 60),
            epochs=data.get("epochs", 50),
            hidden_layer_size=data.get("hidden_layer_size", 50),
            num_layers=data.get("num_layers", 2),
            learning_rate=data.get("learning_rate", 0.001),
            batch_size=data.get("batch_size", 32),
            test_split_size=data.get("test_split_size", 0.2),
            additional_params=data.get("additional_params", {}),
            user_id=data.get("user_id"),
            created_at=data.get("created_at"),
        )
    
    def update_status(self, status: Union[TaskStatus, str]) -> None:
        """Update the task status."""
        self.status = status if isinstance(status, str) else status.value
        self.updated_at = datetime.now().isoformat()
    
    def set_result(self, result: Dict[str, Any]) -> None:
        """Set the task result."""
        self.result = result
        self.status = TaskStatus.COMPLETED.value
        self.updated_at = datetime.now().isoformat()
    
    def set_error(self, error: str) -> None:
        """Set the task error."""
        self.error = error
        self.status = TaskStatus.FAILED.value
        self.updated_at = datetime.now().isoformat()


async def submit_model_training_task(
    task: ModelTrainingTask,
    rabbitmq_client: Optional[AsyncRabbitMQClient] = None,
) -> str:
    """Submit a model training task to the queue asynchronously.
    
    Args:
        task: The model training task to submit.
        rabbitmq_client: Optional client to use. If None, a new client is created.
        
    Returns:
        str: The task ID.
    """
    # Create a client if one wasn't provided
    should_close_client = False
    if rabbitmq_client is None:
        rabbitmq_client = AsyncRabbitMQClient()
        await rabbitmq_client.connect()
        should_close_client = True
    
    # Submit the task
    task_dict = task.to_dict()
    success = await rabbitmq_client.publish_message(
        MODEL_TRAINING_QUEUE,
        task_dict,
        correlation_id=task.task_id
    )
    
    # Close the client if we created it
    if should_close_client:
        await rabbitmq_client.close()
    
    if not success:
        raise Exception("Failed to submit model training task")
    
    logger.info(f"Submitted model training task: {task.task_id}")
    return task.task_id


def submit_model_training_task_sync(
    task: ModelTrainingTask,
    rabbitmq_client: Optional[RabbitMQClient] = None,
) -> str:
    """Submit a model training task to the queue synchronously.
    
    Args:
        task: The model training task to submit.
        rabbitmq_client: Optional client to use. If None, a new client is created.
        
    Returns:
        str: The task ID.
    """
    # Create a client if one wasn't provided
    should_close_client = False
    if rabbitmq_client is None:
        rabbitmq_client = RabbitMQClient()
        rabbitmq_client.connect()
        should_close_client = True
    
    # Submit the task
    task_dict = task.to_dict()
    success = rabbitmq_client.publish_message(
        MODEL_TRAINING_QUEUE,
        task_dict,
        correlation_id=task.task_id
    )
    
    # Close the client if we created it
    if should_close_client:
        rabbitmq_client.close()
    
    if not success:
        raise Exception("Failed to submit model training task")
    
    logger.info(f"Submitted model training task: {task.task_id}")
    return task.task_id


async def send_task_result_notification(
    task_id: str,
    status: Union[TaskStatus, str],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    rabbitmq_client: Optional[AsyncRabbitMQClient] = None,
) -> bool:
    """Send a notification about a task result asynchronously.
    
    Args:
        task_id: The ID of the task.
        status: The status of the task.
        result: Optional result data.
        error: Optional error message.
        rabbitmq_client: Optional client to use. If None, a new client is created.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Create a client if one wasn't provided
    should_close_client = False
    if rabbitmq_client is None:
        rabbitmq_client = AsyncRabbitMQClient()
        await rabbitmq_client.connect()
        should_close_client = True
    
    # Create the notification
    notification = {
        "task_id": task_id,
        "status": status if isinstance(status, str) else status.value,
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "error": error,
    }
    
    # Send the notification
    success = await rabbitmq_client.publish_message(
        RESULT_NOTIFICATION_QUEUE,
        notification,
        correlation_id=task_id
    )
    
    # Close the client if we created it
    if should_close_client:
        await rabbitmq_client.close()
    
    if not success:
        logger.error(f"Failed to send task result notification for task: {task_id}")
        return False
    
    logger.info(f"Sent task result notification for task: {task_id}")
    return True


def send_task_result_notification_sync(
    task_id: str,
    status: Union[TaskStatus, str],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    rabbitmq_client: Optional[RabbitMQClient] = None,
) -> bool:
    """Send a notification about a task result synchronously.
    
    Args:
        task_id: The ID of the task.
        status: The status of the task.
        result: Optional result data.
        error: Optional error message.
        rabbitmq_client: Optional client to use. If None, a new client is created.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Create a client if one wasn't provided
    should_close_client = False
    if rabbitmq_client is None:
        rabbitmq_client = RabbitMQClient()
        rabbitmq_client.connect()
        should_close_client = True
    
    # Create the notification
    notification = {
        "task_id": task_id,
        "status": status if isinstance(status, str) else status.value,
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "error": error,
    }
    
    # Send the notification
    success = rabbitmq_client.publish_message(
        RESULT_NOTIFICATION_QUEUE,
        notification,
        correlation_id=task_id
    )
    
    # Close the client if we created it
    if should_close_client:
        rabbitmq_client.close()
    
    if not success:
        logger.error(f"Failed to send task result notification for task: {task_id}")
        return False
    
    logger.info(f"Sent task result notification for task: {task_id}")
    return True 