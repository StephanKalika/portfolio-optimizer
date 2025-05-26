#!/usr/bin/env python3
"""
Worker script for processing model training tasks from RabbitMQ queue.

This script runs as a separate process from the API server and consumes
messages from the model_training_tasks queue.
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import aio_pika
import requests
from aio_pika.abc import AbstractIncomingMessage

from shared_modules.message_queue.model_tasks import (
    ModelTrainingTask,
    TaskStatus,
    RESULT_NOTIFICATION_QUEUE,
    send_task_result_notification,
)
from shared_modules.message_queue.rabbitmq import (
    AsyncRabbitMQClient,
    MODEL_TRAINING_QUEUE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("worker.log"),
    ],
)
logger = logging.getLogger("model_training_worker")

# Model Training Service URL
MODEL_TRAINING_SERVICE_URL = os.getenv(
    "MODEL_TRAINING_SERVICE_URL", "http://model_training_service:8000"
)


async def process_training_task(
    task: ModelTrainingTask, rabbitmq_client: AsyncRabbitMQClient
) -> None:
    """Process a model training task by sending it to the Model Training Service.
    
    Args:
        task: The model training task to process.
        rabbitmq_client: Client for sending notifications.
    """
    logger.info(f"Processing model training task: {task.task_id}")
    
    # Update task status to RUNNING
    task.update_status(TaskStatus.RUNNING)
    await send_task_result_notification(
        task.task_id, TaskStatus.RUNNING, rabbitmq_client=rabbitmq_client
    )
    
    try:
        # Prepare the payload for the Model Training Service
        payload = {
            "tickers": task.tickers,
            "model_name": task.model_name,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "model_type": task.model_type,
            "sequence_length": task.sequence_length,
            "epochs": task.epochs,
            "hidden_layer_size": task.hidden_layer_size,
            "num_layers": task.num_layers,
            "learning_rate": task.learning_rate,
            "batch_size": task.batch_size,
            "test_split_size": task.test_split_size,
        }
        
        # Add any additional params
        if task.additional_params:
            for key, value in task.additional_params.items():
                if key not in payload:
                    payload[key] = value
        
        # Send the request to the Model Training Service
        logger.info(f"Sending training request to Model Training Service for task: {task.task_id}")
        train_url = f"{MODEL_TRAINING_SERVICE_URL}/model/train"
        
        # Model training can take a long time, use a long timeout
        response = requests.post(train_url, json=payload, timeout=3600)
        
        if response.status_code == 200 or response.status_code == 201:
            response_data = response.json()
            logger.info(f"Model training completed successfully for task: {task.task_id}")
            
            # Set the task result and notify
            task.set_result(response_data)
            await send_task_result_notification(
                task.task_id, TaskStatus.COMPLETED, result=response_data, rabbitmq_client=rabbitmq_client
            )
        else:
            # If the request failed, set the error and notify
            error_data = response.json() if response.headers.get("content-type") == "application/json" else {"error": response.text}
            error_message = f"Model training failed with status code: {response.status_code}, error: {error_data}"
            logger.error(error_message)
            
            task.set_error(error_message)
            await send_task_result_notification(
                task.task_id, TaskStatus.FAILED, error=error_message, rabbitmq_client=rabbitmq_client
            )
    except Exception as e:
        # Handle any exceptions during processing
        error_message = f"Error processing model training task: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        task.set_error(error_message)
        await send_task_result_notification(
            task.task_id, TaskStatus.FAILED, error=error_message, rabbitmq_client=rabbitmq_client
        )


async def message_handler(
    message: AbstractIncomingMessage, rabbitmq_client: AsyncRabbitMQClient
) -> None:
    """Handle incoming messages from the queue.
    
    Args:
        message: The incoming message from RabbitMQ.
        rabbitmq_client: Client for sending notifications.
    """
    async with message.process():
        logger.info(f"Received message with correlation_id: {message.correlation_id}")
        
        try:
            # Parse the message body
            body = message.body.decode()
            data = json.loads(body)
            
            # Create a task object
            task = ModelTrainingTask.from_dict(data)
            
            # Process the task
            await process_training_task(task, rabbitmq_client)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.error(traceback.format_exc())


async def run_worker() -> None:
    """Run the worker process to consume messages from the queue."""
    logger.info("Starting model training worker...")
    
    # Create RabbitMQ client
    rabbitmq_client = AsyncRabbitMQClient()
    
    try:
        # Connect to RabbitMQ
        await rabbitmq_client.connect()
        logger.info("Connected to RabbitMQ")
        
        # Get the queue
        queue = await rabbitmq_client.channel.declare_queue(
            MODEL_TRAINING_QUEUE, durable=True
        )
        
        # Set up the consumer with a wrapper function
        async def handle_message(message: AbstractIncomingMessage) -> None:
            await message_handler(message, rabbitmq_client)
        
        # Start consuming messages
        await queue.consume(handle_message)
        logger.info(f"Started consuming messages from queue: {MODEL_TRAINING_QUEUE}")
        
        # Keep the worker running
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in worker: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Close the connection when done
        await rabbitmq_client.close()
        logger.info("Worker stopped")


if __name__ == "__main__":
    try:
        # Run the worker
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped by keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception in worker: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 