# settings_manager.py - Secure LLM Settings Manager
# Handles encrypted storage of API keys and LLM configuration

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("settings_manager")

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not installed. API keys will be stored in plaintext!")

# Settings directory
SETTINGS_DIR = Path.home() / ".agentest"
SETTINGS_FILE = SETTINGS_DIR / "llm_settings.json"
KEY_FILE = SETTINGS_DIR / ".key"


class SecureSettingsManager:
    """
    Manages LLM settings with encrypted API key storage.
    
    Settings are stored in ~/.agentest/llm_settings.json
    Encryption key is stored in ~/.agentest/.key
    """
    
    # Available providers
    PROVIDERS = [
        {"id": "local", "name": "Local (Qwen)", "requires_key": False, "default_model": "qwen2.5:7b-instruct-q6_k"},
        {"id": "ollama", "name": "Ollama", "requires_key": False},
        {"id": "lmstudio", "name": "LM Studio", "requires_key": False},
        {"id": "openrouter", "name": "OpenRouter", "requires_key": True},
        {"id": "anthropic", "name": "Anthropic Claude", "requires_key": True},
        {"id": "gemini", "name": "Google Gemini", "requires_key": True},
        {"id": "openai", "name": "OpenAI", "requires_key": True},
        {"id": "custom", "name": "Custom (OpenAI-compatible)", "requires_key": False},
    ]
    
    # Default settings, if ui and env variables are not set
    DEFAULT_SETTINGS = {
        "provider": "local",
        "base_url": "http://localhost:11434",
        "api_key": "",
        "model": "qwen2.5:7b-instruct-q6_k",
    }
    
    def __init__(self):
        self._ensure_settings_dir()
        self._fernet: Optional[Fernet] = None
        if CRYPTO_AVAILABLE:
            self._init_encryption()
    
    def _ensure_settings_dir(self):
        """Create settings directory if it doesn't exist"""
        if not SETTINGS_DIR.exists():
            SETTINGS_DIR.mkdir(parents=True, mode=0o700)
            logger.info(f"Created settings directory: {SETTINGS_DIR}")
    
    def _init_encryption(self):
        """Initialize Fernet encryption key"""
        if KEY_FILE.exists():
            # Load existing key
            key = KEY_FILE.read_bytes()
        else:
            # Generate new key
            key = Fernet.generate_key()
            KEY_FILE.write_bytes(key)
            # Restrict key file permissions (Windows doesn't have chmod)
            if os.name != 'nt':
                KEY_FILE.chmod(0o600)
            logger.info("Generated new encryption key")
        
        self._fernet = Fernet(key)
    
    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string"""
        if not plaintext:
            return ""
        if self._fernet:
            return self._fernet.encrypt(plaintext.encode()).decode()
        return plaintext  # No encryption available
    
    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string"""
        if not ciphertext:
            return ""
        if self._fernet:
            try:
                return self._fernet.decrypt(ciphertext.encode()).decode()
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return ""
        return ciphertext  # No encryption available
    
    def _mask_api_key(self, api_key: str) -> str:
        """Mask API key for display (show only last 4 chars)"""
        if not api_key or len(api_key) < 8:
            return "••••••••" if api_key else ""
        return f"••••••••{api_key[-4:]}"
    
    def _load_raw(self) -> Dict[str, Any]:
        """Load raw settings from file"""
        if not SETTINGS_FILE.exists():
            return self.DEFAULT_SETTINGS.copy()
        
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return self.DEFAULT_SETTINGS.copy()
    
    def get_settings(self, include_raw_key: bool = False) -> Dict[str, Any]:
        """
        Get current settings.
        
        Args:
            include_raw_key: If True, include decrypted API key (for backend use only)
                           If False, include masked API key (for UI display)
        
        Returns:
            Settings dictionary
        """
        raw = self._load_raw()
        
        # Decrypt API key
        encrypted_key = raw.get("api_key_encrypted", "")
        decrypted_key = self._decrypt(encrypted_key) if encrypted_key else raw.get("api_key", "")
        
        settings = {
            "provider": raw.get("provider", self.DEFAULT_SETTINGS["provider"]),
            "base_url": raw.get("base_url", self.DEFAULT_SETTINGS["base_url"]),
            "model": raw.get("model", self.DEFAULT_SETTINGS["model"]),
        }
        
        if include_raw_key:
            settings["api_key"] = decrypted_key
        else:
            settings["api_key_masked"] = self._mask_api_key(decrypted_key)
            settings["has_api_key"] = bool(decrypted_key)
        
        return settings
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Save settings with encrypted API key.
        
        Args:
            settings: Dictionary containing provider, base_url, api_key (optional), model
        
        Returns:
            True if successful
        """
        try:
            # Prepare data for storage
            data = {
                "provider": settings.get("provider", self.DEFAULT_SETTINGS["provider"]),
                "base_url": settings.get("base_url", self.DEFAULT_SETTINGS["base_url"]),
                "model": settings.get("model", self.DEFAULT_SETTINGS["model"]),
            }
            
            # Handle API key
            api_key = settings.get("api_key", "")
            if api_key:
                # Only update if a new key is provided
                data["api_key_encrypted"] = self._encrypt(api_key)
            else:
                # Preserve existing encrypted key if no new key provided
                existing = self._load_raw()
                if "api_key_encrypted" in existing:
                    data["api_key_encrypted"] = existing["api_key_encrypted"]
            
            # Write to file
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Settings saved to {SETTINGS_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test LLM connection with current settings.
        
        Returns:
            (success, message) tuple
        """
        import httpx
        
        settings = self.get_settings(include_raw_key=True)
        provider = settings["provider"]
        base_url = settings["base_url"]
        api_key = settings.get("api_key", "")
        model = settings["model"]
        
        try:
            if provider in ("local", "ollama"):
                # Test Ollama connection (local also uses Ollama API)
                resp = httpx.get(f"{base_url}/api/tags", timeout=10.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    if model in model_names or any(model in n for n in model_names):
                        return True, f"Connected to Ollama. Model '{model}' available."
                    return True, f"Connected to Ollama. Available models: {', '.join(model_names[:5])}"
                return False, f"Ollama responded with status {resp.status_code}"
            
            elif provider in ("openrouter", "openai", "lmstudio", "custom"):
                # Test OpenAI-compatible API
                url = base_url.rstrip("/") + "/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = httpx.get(url, headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    return True, f"Connected to {provider}. API is responding."
                elif resp.status_code == 401:
                    return False, "Authentication failed. Check your API key."
                return False, f"API responded with status {resp.status_code}"
            
            elif provider == "anthropic":
                # For Anthropic, we just verify the key format
                if not api_key:
                    return False, "Anthropic requires an API key"
                if not api_key.startswith("sk-ant-"):
                    return False, "Invalid Anthropic API key format (should start with sk-ant-)"
                return True, "Anthropic API key format looks valid"
            
            elif provider == "gemini":
                # Test Gemini connection
                if not api_key:
                    return False, "Gemini requires an API key"
                # Simple validation - actual model list requires SDK
                return True, "Gemini API key configured"
            
            else:
                return False, f"Unknown provider: {provider}"
                
        except httpx.ConnectError:
            return False, f"Could not connect to {base_url}. Is the server running?"
        except httpx.TimeoutException:
            return False, f"Connection to {base_url} timed out"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def get_providers(self) -> list:
        """Get list of available providers for UI dropdown"""
        return self.PROVIDERS.copy()


_settings_manager: Optional[SecureSettingsManager] = None


def get_settings_manager() -> SecureSettingsManager:
    """Get or create the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SecureSettingsManager()
    return _settings_manager
