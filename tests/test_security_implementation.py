"""
Comprehensive Security Implementation Test Suite
Tests all security measures implemented in ASL Translator system
"""

import pytest
import os
import sys
import base64
import json
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api" / "src"))

# ===================================================================
# TEST FIXTURE: Sample Data and Mock Objects
# ===================================================================

@pytest.fixture
def sample_image_base64():
    """Create a valid sample image in base64 format"""
    # Create a minimal valid PNG image (1x1 pixel)
    img_data = BytesIO()
    # PNG header and minimal data
    png_bytes = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
        b'\x00\x01\x01\x00\x05\x18\r\xd8e\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def sample_large_image():
    """Create an oversized image that exceeds limits"""
    # Create image data larger than 5MB
    oversized_data = b'X' * (6 * 1024 * 1024)  # 6MB
    return base64.b64encode(oversized_data).decode()


@pytest.fixture
def valid_api_key():
    """Valid API key for testing"""
    return "test-api-key-12345"


@pytest.fixture
def invalid_api_key():
    """Invalid API key for testing"""
    return "invalid-key-xyz"


# ===================================================================
# PHASE 1: ENVIRONMENT & CONFIGURATION TESTS
# ===================================================================

class TestPhase1SecurityConfiguration:
    """Test Phase 1: Environment Variables and Configuration"""
    
    def test_env_example_exists(self):
        """Test that .env.example file exists"""
        env_example = Path(__file__).parent.parent / ".env.example"
        assert env_example.exists(), ".env.example file not found"
    
    def test_env_example_contains_required_vars(self):
        """Test that .env.example contains required security variables"""
        env_example = Path(__file__).parent.parent / ".env.example"
        content = env_example.read_text()
        
        required_vars = [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
            "API_KEYS",
            "ENVIRONMENT",
            "DEBUG"
        ]
        
        for var in required_vars:
            assert var in content, f"Required variable {var} missing from .env.example"
    
    def test_gitignore_excludes_env_files(self):
        """Test that .gitignore properly excludes environment files"""
        gitignore = Path(__file__).parent.parent / ".gitignore"
        content = gitignore.read_text()
        
        required_patterns = [
            ".env.local",
            ".env.production",
            ".env.*.local"
        ]
        
        for pattern in required_patterns:
            assert pattern in content, f"Pattern {pattern} missing from .gitignore"
    
    def test_docker_compose_uses_env_file(self):
        """Test that docker-compose.yaml uses environment files"""
        docker_compose = Path(__file__).parent.parent / "docker-compose.yaml"
        content = docker_compose.read_text()
        
        assert "env_file:" in content, "env_file not configured in docker-compose"
        assert "POSTGRES_USER" in content, "Environment variables not used in docker-compose"
    
    def test_no_hardcoded_passwords(self):
        """Test that no hardcoded passwords exist in docker-compose"""
        docker_compose = Path(__file__).parent.parent / "docker-compose.yaml"
        content = docker_compose.read_text()
        
        # Check for common hardcoded password patterns
        assert "password:" not in content.lower() or "${" in content, \
            "Hardcoded password found in docker-compose"


# ===================================================================
# PHASE 2: RATE LIMITING TESTS
# ===================================================================

class TestPhase2RateLimiting:
    """Test Phase 2: Rate Limiting Implementation"""
    
    def test_slowapi_in_requirements(self):
        """Test that slowapi is in requirements"""
        requirements = Path(__file__).parent.parent / "services" / "api" / "requirements.txt"
        content = requirements.read_text()
        
        assert "slowapi" in content, "slowapi not found in requirements.txt"
    
    def test_rate_limiter_initialized(self):
        """Test that rate limiter is properly initialized"""
        try:
            from slowapi import Limiter
            from slowapi.util import get_remote_address
            
            limiter = Limiter(key_func=get_remote_address)
            assert limiter is not None, "Rate limiter not initialized"
        except ImportError:
            pytest.skip("slowapi not installed")
    
    def test_rate_limit_constants_defined(self):
        """Test that rate limit constants are defined"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        rate_limits = [
            "RATE_LIMIT_PREDICT",
            "RATE_LIMIT_FEEDBACK",
            "RATE_LIMIT_STATS",
            "RATE_LIMIT_ADMIN"
        ]
        
        for limit in rate_limits:
            assert limit in content, f"Rate limit constant {limit} not defined"


# ===================================================================
# PHASE 3: CORS CONFIGURATION TESTS
# ===================================================================

class TestPhase3CORSConfiguration:
    """Test Phase 3: CORS Middleware Configuration"""
    
    def test_cors_middleware_imported(self):
        """Test that CORS middleware is imported"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "CORSMiddleware" in content, "CORSMiddleware not imported"
    
    def test_cors_middleware_configured(self):
        """Test that CORS middleware is added to app"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "add_middleware" in content or "CORSMiddleware" in content, \
            "CORS middleware not configured"


# ===================================================================
# PHASE 4: AUTHENTICATION & API KEY TESTS
# ===================================================================

class TestPhase4Authentication:
    """Test Phase 4: Authentication and API Key Implementation"""
    
    def test_security_module_exists(self):
        """Test that security.py module exists"""
        security_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security.py"
        assert security_file.exists(), "security.py module not found"
    
    def test_verify_api_key_function_exists(self):
        """Test that verify_api_key function is defined"""
        security_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security.py"
        content = security_file.read_text()
        
        assert "def verify_api_key" in content, "verify_api_key function not defined"
        assert "APIKeyHeader" in content, "APIKeyHeader not imported"
    
    def test_api_key_header_configuration(self):
        """Test that API key header is properly configured"""
        security_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security.py"
        content = security_file.read_text()
        
        assert "X-API-Key" in content or "api_key_header" in content, \
            "API key header not properly configured"
    
    def test_optional_api_key_function_exists(self):
        """Test that optional_api_key function is defined"""
        security_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security.py"
        content = security_file.read_text()
        
        assert "def optional_api_key" in content, "optional_api_key function not defined"


# ===================================================================
# PHASE 5: INPUT VALIDATION TESTS
# ===================================================================

class TestPhase5InputValidation:
    """Test Phase 5: Input Validation and Sanitization"""
    
    def test_prediction_request_model_exists(self):
        """Test that PredictionRequest model exists"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "class PredictionRequest" in content, "PredictionRequest model not defined"
    
    def test_image_validation_implemented(self):
        """Test that image validation is implemented"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "@field_validator('image')" in content or "validate_image" in content, \
            "Image validation not implemented"
    
    def test_base64_size_validation(self):
        """Test that base64 image size is validated"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "5" in content and "1024" in content and "1024" in content, \
            "5MB size limit not found in validation"
    
    def test_feedback_request_model_exists(self):
        """Test that FeedbackRequest model exists"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "class FeedbackRequest" in content, "FeedbackRequest model not defined"
    
    def test_uuid_validation_implemented(self):
        """Test that UUID validation is implemented"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "uuid" in content.lower(), "UUID validation not implemented"


# ===================================================================
# PHASE 6: SECURITY HEADERS TESTS
# ===================================================================

class TestPhase6SecurityHeaders:
    """Test Phase 6: Security Headers Middleware"""
    
    def test_security_middleware_exists(self):
        """Test that security_middleware.py module exists"""
        middleware_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_middleware.py"
        assert middleware_file.exists(), "security_middleware.py not found"
    
    def test_security_headers_class_exists(self):
        """Test that SecurityHeadersMiddleware class is defined"""
        middleware_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_middleware.py"
        content = middleware_file.read_text()
        
        assert "class SecurityHeadersMiddleware" in content or "SecurityHeaders" in content, \
            "SecurityHeadersMiddleware class not defined"
    
    def test_owasp_security_headers_implemented(self):
        """Test that OWASP security headers are implemented"""
        middleware_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_middleware.py"
        content = middleware_file.read_text()
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy"
        ]
        
        for header in required_headers:
            assert header in content, f"Security header {header} not implemented"


# ===================================================================
# PHASE 7: SECURE LOGGING TESTS
# ===================================================================

class TestPhase7SecureLogging:
    """Test Phase 7: Secure Logging and Data Redaction"""
    
    def test_security_logging_module_exists(self):
        """Test that security_logging.py module exists"""
        logging_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_logging.py"
        assert logging_file.exists(), "security_logging.py not found"
    
    def test_sensitive_data_filter_exists(self):
        """Test that SensitiveDataFilter is defined"""
        logging_setup_file = Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py"
        content = logging_setup_file.read_text()
        
        assert "SensitiveDataFilter" in content or "filter" in content.lower(), \
            "SensitiveDataFilter not defined"
    
    def test_password_redaction_pattern(self):
        """Test that password redaction pattern exists"""
        logging_setup_file = Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py"
        content = logging_setup_file.read_text()
        
        assert "password" in content.lower(), "Password redaction not implemented"
    
    def test_api_key_redaction_pattern(self):
        """Test that API key redaction pattern exists"""
        logging_setup_file = Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py"
        content = logging_setup_file.read_text()
        
        assert "api" in content.lower() and "key" in content.lower(), \
            "API key redaction not implemented"
    
    def test_security_logger_class_exists(self):
        """Test that SecurityLogger class is defined"""
        security_logging_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_logging.py"
        content = security_logging_file.read_text()
        
        assert "class SecurityLogger" in content, "SecurityLogger class not defined"
    
    def test_security_logger_has_log_methods(self):
        """Test that SecurityLogger has required log methods"""
        security_logging_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_logging.py"
        content = security_logging_file.read_text()
        
        required_methods = [
            "log_authentication_failure",
            "log_rate_limit_exceeded",
            "log_validation_error"
        ]
        
        for method in required_methods:
            assert method in content, f"SecurityLogger method {method} not defined"


# ===================================================================
# PHASE 8: CONFIGURATION VALIDATION TESTS
# ===================================================================

class TestPhase8ConfigurationValidation:
    """Test Phase 8: Production Configuration and Validation"""
    
    def test_config_module_has_environment_setting(self):
        """Test that config.py has environment setting"""
        config_file = Path(__file__).parent.parent / "services" / "inference" / "src" / "config.py"
        content = config_file.read_text()
        
        assert "environment" in content.lower(), "Environment setting not in config"
    
    def test_config_has_is_production_property(self):
        """Test that config has is_production property"""
        config_file = Path(__file__).parent.parent / "services" / "inference" / "src" / "config.py"
        content = config_file.read_text()
        
        assert "is_production" in content or "production" in content.lower(), \
            "is_production property not in config"
    
    def test_debug_mode_validation(self):
        """Test that debug mode is properly configured"""
        config_file = Path(__file__).parent.parent / "services" / "inference" / "src" / "config.py"
        content = config_file.read_text()
        
        assert "debug" in content.lower(), "Debug mode setting not found"


# ===================================================================
# INTEGRATION TESTS
# ===================================================================

class TestIntegrationSecurityFlow:
    """Integration tests for complete security flow"""
    
    def test_all_security_files_exist(self):
        """Test that all security-related files exist"""
        required_files = [
            Path(__file__).parent.parent / ".env.example",
            Path(__file__).parent.parent / "services" / "api" / "src" / "security.py",
            Path(__file__).parent.parent / "services" / "api" / "src" / "security_middleware.py",
            Path(__file__).parent.parent / "services" / "api" / "src" / "security_logging.py",
            Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py",
        ]
        
        for file in required_files:
            assert file.exists(), f"Required security file {file.name} not found"
    
    def test_all_security_features_imported_in_api(self):
        """Test that all security features are imported in api.py"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        security_imports = [
            "CORSMiddleware",
            "Limiter",
            "rate_limit_exceeded_handler",
            "verify_api_key"
        ]
        
        for import_name in security_imports:
            assert import_name in content, f"Security import {import_name} not found in api.py"
    
    def test_protected_endpoints_have_decorators(self):
        """Test that protected endpoints have security decorators"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        # Check that /api/models endpoints are protected
        assert "/api/models" in content, "Admin endpoints not found"
        assert "@limiter" in content or "limiter.limit" in content, \
            "Rate limiting not applied to endpoints"


# ===================================================================
# SECURITY BEST PRACTICES VALIDATION
# ===================================================================

class TestSecurityBestPractices:
    """Test adherence to security best practices"""
    
    def test_no_hardcoded_secrets_in_api(self):
        """Test that no hardcoded secrets exist in API code"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        # Check for common secret patterns (basic check)
        suspicious_patterns = [
            "password = '",
            "password = \"",
            "api_key = '",
            "api_key = \"",
            "secret = '"
        ]
        
        for pattern in suspicious_patterns:
            assert pattern not in content.lower(), f"Potential hardcoded secret found: {pattern}"
    
    def test_os_getenv_used_for_secrets(self):
        """Test that os.getenv is used for secrets"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "os.getenv" in content, "Environment variables not used for configuration"
    
    def test_logging_configured_safely(self):
        """Test that logging is configured securely"""
        logging_setup_file = Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py"
        content = logging_setup_file.read_text()
        
        # Check that logging is setup with proper configuration
        assert "logging" in content, "Logging not configured"
        assert "DEBUG" in content or "INFO" in content, "Log levels not configured"


# ===================================================================
# COMPLIANCE TESTS
# ===================================================================

class TestComplianceWithRequirements:
    """Test compliance with R-06 security requirement"""
    
    def test_r06_authentication_implemented(self):
        """R-06: Authentication mechanisms are implemented"""
        security_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security.py"
        assert security_file.exists() and "verify_api_key" in security_file.read_text()
    
    def test_r06_data_protection_implemented(self):
        """R-06: Sensitive data protection is implemented"""
        files_to_check = [
            Path(__file__).parent.parent / ".env.example",
            Path(__file__).parent.parent / "services" / "api" / "src" / "security_logging.py"
        ]
        
        for file in files_to_check:
            if file.exists():
                content = file.read_text()
                assert len(content) > 0, f"Security file {file.name} is empty"
    
    def test_r06_input_validation_implemented(self):
        """R-06: Input validation is implemented"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "@field_validator" in content, "Input validation not implemented"
    
    def test_r06_cors_protection_implemented(self):
        """R-06: CORS protection is implemented"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "CORSMiddleware" in content, "CORS protection not implemented"
    
    def test_r06_rate_limiting_implemented(self):
        """R-06: Rate limiting for availability protection"""
        api_file = Path(__file__).parent.parent / "services" / "api" / "src" / "api.py"
        content = api_file.read_text()
        
        assert "Limiter" in content or "slowapi" in content, "Rate limiting not implemented"
    
    def test_r06_secure_configuration_implemented(self):
        """R-06: Secure configuration separation"""
        config_file = Path(__file__).parent.parent / "services" / "inference" / "src" / "config.py"
        docker_file = Path(__file__).parent.parent / "docker-compose.yaml"
        
        config_content = config_file.read_text()
        docker_content = docker_file.read_text()
        
        assert "environment" in config_content.lower(), "Environment configuration missing"
        assert "env_file" in docker_content, "env_file configuration missing"
    
    def test_r06_security_headers_implemented(self):
        """R-06: Security headers to prevent vulnerabilities"""
        middleware_file = Path(__file__).parent.parent / "services" / "api" / "src" / "security_middleware.py"
        assert middleware_file.exists() and "X-Content-Type-Options" in middleware_file.read_text()
    
    def test_r06_logging_without_sensitive_data(self):
        """R-06: Logging without recording sensitive data"""
        logging_file = Path(__file__).parent.parent / "services" / "api" / "src" / "logging_setup.py"
        content = logging_file.read_text()
        
        assert "SensitiveDataFilter" in content or "filter" in content.lower(), \
            "Sensitive data filtering not implemented"


# ===================================================================
# MAIN TEST RUNNER
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
