"""
Firebase/Firestore initialization and configuration.
Architectural Choice: Singleton pattern ensures single Firestore connection
across entire system, preventing connection exhaustion and maintaining
consistent error handling.
"""
import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.client import Client
from google.cloud.firestore_v1.base_client import BaseClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation."""
    project_id: str
    service_account_path: Optional[str] = None
    use_default_credentials: bool = False
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if not self.project_id:
            raise ValueError("Firebase project_id is required")
        
        if not self.use_default_credentials and not self.service_account_path:
            # Check environment variable as fallback
            env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if env_path and Path(env_path).exists():
                self.service_account_path = env_path
            else:
                raise ValueError(
                    "Either service_account_path must be provided, "
                    "GOOGLE_APPLICATION_CREDENTIALS env var must be set, "
                    "or use_default_credentials must be True"
                )
        
        if self.service_account_path and not Path(self.service_account_path).exists():
            raise FileNotFoundError(
                f"Service account file not found: {self.service_account_path}"
            )

class FirebaseManager:
    """Singleton manager for Firebase/Firestore connections."""
    _instance: Optional['FirebaseManager'] = None
    _client: Optional[Client] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, config: FirebaseConfig) -> Client:
        """
        Initialize Firebase connection with robust error handling.
        
        Args:
            config: Validated Firebase configuration
            
        Returns:
            Firestore client instance
            
        Raises:
            RuntimeError: If Firebase initialization fails
            ValueError: If configuration is invalid
        """
        if self._initialized and self._client:
            logger.info("Firebase already initialized, returning existing client")
            return self._client
        
        try:
            # Check for existing Firebase app to prevent duplicate initialization
            if not firebase_admin._apps:
                if config.use_default_credentials:
                    logger.info("Initializing Firebase with default credentials")
                    cred = credentials.ApplicationDefault()
                else:
                    logger.info(f"Initializing Firebase with service account: {config.service_account_path}")
                    cred = credentials.Certificate(config.service_account_path)
                
                firebase_admin.initialize_app(cred, {
                    'projectId': config.project_id
                })
            else:
                logger.info("Using existing Firebase app")
            
            # Initialize Firestore client
            self._client = firestore.client()
            self._initialized = True
            
            # Test connection
            test_doc = self._client.collection('_connection_tests').document('test')
            test_doc.set({'timestamp': firestore.SERVER_TIMESTAMP})
            test_doc.delete()
            
            logger.info(f"Firebase Firestore initialized successfully for project: {config.project_id}")
            return self._client
            
        except Exception as e:
            error_msg = f"Firebase initialization failed: {str(e)}"
            logger.error(error_msg)
            # Reset state on failure
            self._initialized = False
            self._client = None
            raise RuntimeError(error_msg) from e
    
    def get_client(self) -> Client:
        """
        Get Firestore client, ensuring initialization.
        
        Returns:
            Firestore client
            
        Raises:
            RuntimeError: If Firebase is not initialized
        """
        if not self._initialized or not self._client:
            raise RuntimeError(
                "Firebase not initialized. Call initialize() first with valid config."
            )
        return self._client
    
    def shutdown(self):
        """Clean shutdown of Firebase connections."""
        try:
            # Firestore client doesn't have explicit close, but we can clean up Firebase app
            firebase_admin.delete_app(firebase_admin.get_app())
            self._client = None
            self._initialized = False
            logger.info("Firebase connections shut down cleanly")
        except Exception as e:
            logger.warning(f"Error during Firebase shutdown: {e}")

# Global instance for easy import
firebase_manager = FirebaseManager()

def initialize_firebase(
    project_id: str,
    service_account_path: Optional[str] = None,
    use_default_credentials: bool = False
) -> Client:
    """
    Convenience function for initializing Firebase.
    
    Args:
        project_id: Google Cloud project ID
        service_account_path: Path to service account JSON file
        use_default_credentials: Use application default credentials
    
    Returns:
        Initialized Firestore client
    """
    config = FirebaseConfig(
        project_id=project_id,
        service_account_path=service_account_path,
        use_default_credentials=use_default_credentials
    )
    return firebase_manager.initialize(config)

def get_firestore_client() -> Client:
    """
    Get the Firestore client instance.
    
    Returns:
        Firestore client
        
    Raises:
        RuntimeError: If Firebase is not initialized
    """
    return firebase_manager.get_client()