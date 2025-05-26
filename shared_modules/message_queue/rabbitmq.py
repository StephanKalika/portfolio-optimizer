"""
RabbitMQ client for Portfolio Optimizer

This module provides functionality for interacting with RabbitMQ
for async message processing across microservices.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import aio_pika
import pika
from pika.exceptions import AMQPConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default RabbitMQ connection parameters
DEFAULT_RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
DEFAULT_RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
DEFAULT_RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
DEFAULT_RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

# Queue names for different services
MODEL_TRAINING_QUEUE = "model_training_tasks"
PORTFOLIO_OPTIMIZATION_QUEUE = "portfolio_optimization_tasks"
RESULT_NOTIFICATION_QUEUE = "result_notifications"

class RabbitMQClient:
    """Client for interacting with RabbitMQ."""
    
    def __init__(
        self,
        host: str = DEFAULT_RABBITMQ_HOST,
        port: int = DEFAULT_RABBITMQ_PORT,
        username: str = DEFAULT_RABBITMQ_USER,
        password: str = DEFAULT_RABBITMQ_PASS,
    ):
        """Initialize the RabbitMQ client with connection parameters."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
        
    def connect(self) -> bool:
        """Establish a connection to RabbitMQ.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare the queues
            self.channel.queue_declare(queue=MODEL_TRAINING_QUEUE, durable=True)
            self.channel.queue_declare(queue=PORTFOLIO_OPTIMIZATION_QUEUE, durable=True)
            self.channel.queue_declare(queue=RESULT_NOTIFICATION_QUEUE, durable=True)
            
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
            return True
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def close(self) -> None:
        """Close the connection to RabbitMQ."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Closed connection to RabbitMQ")
    
    def publish_message(
        self,
        queue_name: str,
        message: Union[Dict[str, Any], str],
        correlation_id: Optional[str] = None,
        priority: int = 0,
    ) -> bool:
        """Publish a message to a specified queue.
        
        Args:
            queue_name: The name of the queue to publish to.
            message: The message to publish (dict will be converted to JSON).
            correlation_id: Optional correlation ID for tracking requests.
            priority: Message priority (0-10, higher is more important).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.connection or not self.channel:
            if not self.connect():
                return False
        
        try:
            # Convert dict to JSON string
            if isinstance(message, dict):
                message = json.dumps(message)
            
            # Set message properties
            properties = pika.BasicProperties(
                content_type='application/json',
                delivery_mode=2,  # Make message persistent
                correlation_id=correlation_id,
                priority=priority
            )
            
            # Publish the message
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=message,
                properties=properties
            )
            
            logger.info(f"Published message to queue: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    def consume_messages(
        self,
        queue_name: str,
        callback: Callable,
        auto_ack: bool = False,
        prefetch_count: int = 1,
    ) -> None:
        """Set up a consumer for processing messages from a queue.
        
        Args:
            queue_name: The name of the queue to consume from.
            callback: Function to call when a message is received.
            auto_ack: Whether to auto-acknowledge messages.
            prefetch_count: Number of messages to prefetch.
        """
        if not self.connection or not self.channel:
            if not self.connect():
                return
        
        # Set up quality of service
        self.channel.basic_qos(prefetch_count=prefetch_count)
        
        # Set up the consumer
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=auto_ack
        )
        
        logger.info(f"Started consuming messages from queue: {queue_name}")
        
        try:
            # Start consuming (this blocks)
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
            self.close()
            logger.info("Stopped consuming messages due to interrupt")
        except Exception as e:
            logger.error(f"Error while consuming messages: {e}")
            self.close()


class AsyncRabbitMQClient:
    """Async client for interacting with RabbitMQ using aio_pika."""
    
    def __init__(
        self,
        host: str = DEFAULT_RABBITMQ_HOST,
        port: int = DEFAULT_RABBITMQ_PORT,
        username: str = DEFAULT_RABBITMQ_USER,
        password: str = DEFAULT_RABBITMQ_PASS,
    ):
        """Initialize the async RabbitMQ client."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
    
    async def connect(self) -> bool:
        """Establish a connection to RabbitMQ asynchronously.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            connection_string = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"
            self.connection = await aio_pika.connect_robust(connection_string)
            self.channel = await self.connection.channel()
            
            # Declare queues
            await self.channel.declare_queue(MODEL_TRAINING_QUEUE, durable=True)
            await self.channel.declare_queue(PORTFOLIO_OPTIMIZATION_QUEUE, durable=True)
            await self.channel.declare_queue(RESULT_NOTIFICATION_QUEUE, durable=True)
            
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port} (async)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ asynchronously: {e}")
            return False
    
    async def close(self) -> None:
        """Close the connection to RabbitMQ asynchronously."""
        if self.connection:
            await self.connection.close()
            logger.info("Closed async connection to RabbitMQ")
    
    async def publish_message(
        self,
        queue_name: str,
        message: Union[Dict[str, Any], str],
        correlation_id: Optional[str] = None,
        priority: int = 0,
    ) -> bool:
        """Publish a message to a specified queue asynchronously.
        
        Args:
            queue_name: The name of the queue to publish to.
            message: The message to publish (dict will be converted to JSON).
            correlation_id: Optional correlation ID for tracking requests.
            priority: Message priority (0-10, higher is more important).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.connection or not self.channel:
            if not await self.connect():
                return False
        
        try:
            # Convert dict to JSON string if needed
            if isinstance(message, dict):
                message = json.dumps(message)
            
            # Create message with properties
            message_body = message.encode() if isinstance(message, str) else message
            message = aio_pika.Message(
                body=message_body,
                content_type='application/json',
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                correlation_id=correlation_id,
                priority=priority
            )
            
            # Publish the message
            await self.channel.default_exchange.publish(
                message,
                routing_key=queue_name
            )
            
            logger.info(f"Published message to queue: {queue_name} (async)")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message asynchronously: {e}")
            return False
    
    async def consume_messages(
        self,
        queue_name: str,
        callback: Callable,
        prefetch_count: int = 1,
    ) -> None:
        """Set up a consumer for processing messages from a queue asynchronously.
        
        Args:
            queue_name: The name of the queue to consume from.
            callback: Async function to call when a message is received.
            prefetch_count: Number of messages to prefetch.
        """
        if not self.connection or not self.channel:
            if not await self.connect():
                return
        
        # Set up quality of service
        await self.channel.set_qos(prefetch_count=prefetch_count)
        
        # Get the queue
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        # Set up the consumer
        await queue.consume(callback)
        
        logger.info(f"Started consuming messages from queue: {queue_name} (async)") 