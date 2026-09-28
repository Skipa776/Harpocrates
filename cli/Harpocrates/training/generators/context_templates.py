"""
Context generation templates for synthetic training data.

Provides positive contexts (secret-like) and negative contexts
(false positive) across multiple programming languages and formats.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Variable names that suggest secrets (positive context)
SECRET_VAR_NAMES = [
    "api_key",
    "API_KEY",
    "apiKey",
    "secret_key",
    "SECRET_KEY",
    "secretKey",
    "access_key",
    "ACCESS_KEY",
    "accessKey",
    "auth_token",
    "AUTH_TOKEN",
    "authToken",
    "password",
    "PASSWORD",
    "passwd",
    "private_key",
    "PRIVATE_KEY",
    "privateKey",
    "credentials",
    "CREDENTIALS",
    "bearer_token",
    "BEARER_TOKEN",
    "bearerToken",
    "secret",
    "SECRET",
    "token",
    "TOKEN",
    "api_secret",
    "API_SECRET",
    "apiSecret",
    "client_secret",
    "CLIENT_SECRET",
    "clientSecret",
    "encryption_key",
    "ENCRYPTION_KEY",
    "encryptionKey",
]

# Variable names that suggest safe content (negative context)
SAFE_VAR_NAMES = [
    "commit_sha",
    "COMMIT_SHA",
    "commitSha",
    "git_hash",
    "GIT_HASH",
    "gitHash",
    "revision",
    "REVISION",
    "rev",
    "REV",
    "file_hash",
    "FILE_HASH",
    "fileHash",
    "checksum",
    "CHECKSUM",
    "digest",
    "DIGEST",
    "fingerprint",
    "FINGERPRINT",
    "uuid",
    "UUID",
    "user_id",
    "USER_ID",
    "userId",
    "request_id",
    "REQUEST_ID",
    "requestId",
    "trace_id",
    "TRACE_ID",
    "traceId",
    "session_id",
    "SESSION_ID",
    "sessionId",
    "content_hash",
    "CONTENT_HASH",
    "contentHash",
    "sha256",
    "SHA256",
    "md5_hash",
    "MD5_HASH",
    "md5Hash",
]

# Test-related variable names (negative context)
TEST_VAR_NAMES = [
    "fake_token",
    "FAKE_TOKEN",
    "fakeToken",
    "mock_secret",
    "MOCK_SECRET",
    "mockSecret",
    "test_api_key",
    "TEST_API_KEY",
    "testApiKey",
    "dummy_password",
    "DUMMY_PASSWORD",
    "dummyPassword",
    "example_key",
    "EXAMPLE_KEY",
    "exampleKey",
    "sample_token",
    "SAMPLE_TOKEN",
    "sampleToken",
    "placeholder",
    "PLACEHOLDER",
]


@dataclass
class ContextTemplate:
    """Template for generating code context."""

    language: str
    file_extension: str
    line_template: str  # Template with {var_name} and {token} placeholders
    context_before: List[str]
    context_after: List[str]
    file_path_template: str


# Python context templates
PYTHON_SECRET_TEMPLATES = [
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["import os", "", "# Configuration"],
        context_after=["", "def connect():", "    pass"],
        file_path_template="config/{name}.py",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template="{var_name} = '{token}'",
        context_before=["from typing import Optional", "", "class Config:"],
        context_after=["", "    def __init__(self):", "        pass"],
        file_path_template="src/config.py",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='os.environ["{var_name}"] = "{token}"',
        context_before=["import os", "import sys", ""],
        context_after=["", "# Setup complete", ""],
        file_path_template="scripts/setup.py",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='config["{var_name}"] = "{token}"',
        context_before=["config = {}", "", "# Load secrets"],
        context_after=["", "return config", ""],
        file_path_template="app/settings.py",
    ),
]

PYTHON_SAFE_TEMPLATES = [
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["import subprocess", "", "# Get git info"],
        context_after=["print(f'Commit: {{commit_sha}}')", "", ""],
        file_path_template="scripts/build.py",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["import hashlib", "", "# Calculate hash"],
        context_after=["return checksum", "", ""],
        file_path_template="utils/hash.py",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["import uuid", "", "def generate_id():"],
        context_after=["    return str(uuid.uuid4())", "", ""],
        file_path_template="utils/ids.py",
    ),
]

# JavaScript/TypeScript context templates
JS_SECRET_TEMPLATES = [
    ContextTemplate(
        language="javascript",
        file_extension=".js",
        line_template='const {var_name} = "{token}";',
        context_before=["const axios = require('axios');", "", "// API Configuration"],
        context_after=["", "module.exports = { apiKey };", ""],
        file_path_template="config/api.js",
    ),
    ContextTemplate(
        language="javascript",
        file_extension=".js",
        line_template='let {var_name} = "{token}";',
        context_before=["'use strict';", "", "// Auth setup"],
        context_after=["", "export default authToken;", ""],
        file_path_template="src/auth.js",
    ),
    ContextTemplate(
        language="typescript",
        file_extension=".ts",
        line_template='const {var_name}: string = "{token}";',
        context_before=["import { Config } from './types';", "", ""],
        context_after=["", "export { apiKey };", ""],
        file_path_template="src/config.ts",
    ),
]

JS_SAFE_TEMPLATES = [
    ContextTemplate(
        language="javascript",
        file_extension=".js",
        line_template='const {var_name} = "{token}";',
        context_before=["const { execSync } = require('child_process');", "", "// Git info"],
        context_after=["console.log(`Build: ${commitSha}`);", "", ""],
        file_path_template="scripts/build.js",
    ),
    ContextTemplate(
        language="javascript",
        file_extension=".js",
        line_template='const {var_name} = "{token}";',
        context_before=["const crypto = require('crypto');", "", "// Hash file"],
        context_after=["return fileHash;", "", ""],
        file_path_template="utils/hash.js",
    ),
]

# YAML context templates
YAML_SECRET_TEMPLATES = [
    ContextTemplate(
        language="yaml",
        file_extension=".yaml",
        line_template="{var_name}: {token}",
        context_before=["# Application configuration", "app:", "  name: myapp"],
        context_after=["", "database:", "  host: localhost"],
        file_path_template="config/settings.yaml",
    ),
    ContextTemplate(
        language="yaml",
        file_extension=".yml",
        line_template='{var_name}: "{token}"',
        context_before=["secrets:", "  # API credentials", ""],
        context_after=["", "  region: us-east-1", ""],
        file_path_template="config/secrets.yml",
    ),
]

YAML_SAFE_TEMPLATES = [
    ContextTemplate(
        language="yaml",
        file_extension=".yaml",
        line_template="{var_name}: {token}",
        context_before=["# Build metadata", "build:", "  version: 1.0.0"],
        context_after=["", "  timestamp: 2024-01-01", ""],
        file_path_template="build/metadata.yaml",
    ),
]

# JSON context templates
JSON_SECRET_TEMPLATES = [
    ContextTemplate(
        language="json",
        file_extension=".json",
        line_template='  "{var_name}": "{token}",',
        context_before=["{", '  "name": "myapp",', ""],
        context_after=["", '  "version": "1.0.0"', "}"],
        file_path_template="config/settings.json",
    ),
]

JSON_SAFE_TEMPLATES = [
    ContextTemplate(
        language="json",
        file_extension=".json",
        line_template='  "{var_name}": "{token}",',
        context_before=["{", '  "build": "production",', ""],
        context_after=["", '  "timestamp": "2024-01-01"', "}"],
        file_path_template="build/info.json",
    ),
]

# .env context templates
ENV_SECRET_TEMPLATES = [
    ContextTemplate(
        language="dotenv",
        file_extension=".env",
        line_template="{var_name}={token}",
        context_before=["# Application secrets", "APP_NAME=myapp", ""],
        context_after=["", "DATABASE_URL=postgres://localhost/db", ""],
        file_path_template=".env",
    ),
    ContextTemplate(
        language="dotenv",
        file_extension=".env",
        line_template='{var_name}="{token}"',
        context_before=["# Production config", "ENV=production", ""],
        context_after=["", "LOG_LEVEL=info", ""],
        file_path_template=".env.production",
    ),
]

# Shell context templates
SHELL_SECRET_TEMPLATES = [
    ContextTemplate(
        language="shell",
        file_extension=".sh",
        line_template='export {var_name}="{token}"',
        context_before=["#!/bin/bash", "", "# Set environment"],
        context_after=["", "echo 'Config loaded'", ""],
        file_path_template="scripts/setup.sh",
    ),
]

SHELL_SAFE_TEMPLATES = [
    ContextTemplate(
        language="shell",
        file_extension=".sh",
        line_template='{var_name}=$(git rev-parse HEAD) # {token}',
        context_before=["#!/bin/bash", "", "# Get git info"],
        context_after=["echo \"Commit: $commit_sha\"", "", ""],
        file_path_template="scripts/build.sh",
    ),
]

# Go context templates
GO_SECRET_TEMPLATES = [
    ContextTemplate(
        language="go",
        file_extension=".go",
        line_template='{var_name} := "{token}"',
        context_before=["package main", "", "import \"os\""],
        context_after=["", "func main() {", "}"],
        file_path_template="cmd/main.go",
    ),
]

GO_SAFE_TEMPLATES = [
    ContextTemplate(
        language="go",
        file_extension=".go",
        line_template='{var_name} := "{token}"',
        context_before=["package git", "", "// GetCommitHash returns the current commit"],
        context_after=["return commitSha", "", "}"],
        file_path_template="internal/git/hash.go",
    ),
]

# Java context templates
JAVA_SECRET_TEMPLATES = [
    ContextTemplate(
        language="java",
        file_extension=".java",
        line_template='private static final String {var_name} = "{token}";',
        context_before=[
            "package com.example.config;",
            "",
            "public class ApiConfig {",
        ],
        context_after=["", "    // API configuration", "}"],
        file_path_template="src/main/java/com/example/config/ApiConfig.java",
    ),
]

JAVA_SAFE_TEMPLATES = [
    ContextTemplate(
        language="java",
        file_extension=".java",
        line_template='String {var_name} = "{token}";',
        context_before=[
            "package com.example.git;",
            "",
            "public class GitInfo {",
        ],
        context_after=["    return commitSha;", "", "}"],
        file_path_template="src/main/java/com/example/git/GitInfo.java",
    ),
]

# Test file context templates (always negative)
TEST_TEMPLATES = [
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["import pytest", "", "def test_auth():"],
        context_after=["    assert fake_token", "", ""],
        file_path_template="tests/test_auth.py",
    ),
    ContextTemplate(
        language="javascript",
        file_extension=".test.js",
        line_template='const {var_name} = "{token}";',
        context_before=["describe('Auth', () => {", "  it('should work', () => {", ""],
        context_after=["    expect(mockSecret).toBeDefined();", "  });", "});"],
        file_path_template="src/__tests__/auth.test.js",
    ),
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='{var_name} = "{token}"',
        context_before=["# Test fixtures", "", "class MockConfig:"],
        context_after=["    pass", "", ""],
        file_path_template="tests/fixtures/config.py",
    ),
]

# Documentation context templates (always negative)
DOC_TEMPLATES = [
    ContextTemplate(
        language="python",
        file_extension=".py",
        line_template='# Example: {var_name} = "{token}"',
        context_before=['"""', "API Authentication Example", ""],
        context_after=["", "Replace with your actual key.", '"""'],
        file_path_template="docs/examples/auth.py",
    ),
    ContextTemplate(
        language="javascript",
        file_extension=".js",
        line_template="// Example: const {var_name} = '{token}';",
        context_before=["/**", " * Authentication setup guide", " */"],
        context_after=["", "// See docs for more info", ""],
        file_path_template="docs/auth.js",
    ),
]


def generate_positive_context(
    token: str,
    var_name: Optional[str] = None,
    language: Optional[str] = None,
    secret_type: Optional[str] = None,
) -> Tuple[str, List[str], List[str], str]:
    """
    Generate positive (secret-like) context for a token.

    Args:
        token: The secret token
        var_name: Optional variable name (random if not provided)
        language: Optional language (random if not provided)
        secret_type: Optional secret type for specialized context

    Returns:
        Tuple of (line_content, context_before, context_after, file_path)
    """
    if var_name is None:
        var_name = random.choice(SECRET_VAR_NAMES)

    # Select templates based on language
    templates = []
    if language == "python":
        templates = PYTHON_SECRET_TEMPLATES
    elif language in ("javascript", "typescript"):
        templates = JS_SECRET_TEMPLATES
    elif language == "yaml":
        templates = YAML_SECRET_TEMPLATES
    elif language == "json":
        templates = JSON_SECRET_TEMPLATES
    elif language == "dotenv":
        templates = ENV_SECRET_TEMPLATES
    elif language == "shell":
        templates = SHELL_SECRET_TEMPLATES
    elif language == "go":
        templates = GO_SECRET_TEMPLATES
    elif language == "java":
        templates = JAVA_SECRET_TEMPLATES
    else:
        # Random language
        all_templates = (
            PYTHON_SECRET_TEMPLATES
            + JS_SECRET_TEMPLATES
            + YAML_SECRET_TEMPLATES
            + JSON_SECRET_TEMPLATES
            + ENV_SECRET_TEMPLATES
            + SHELL_SECRET_TEMPLATES
            + GO_SECRET_TEMPLATES
            + JAVA_SECRET_TEMPLATES
        )
        templates = all_templates

    template = random.choice(templates)

    line_content = template.line_template.format(var_name=var_name, token=token)
    file_path = template.file_path_template.format(name=var_name.lower())

    return (
        line_content,
        template.context_before.copy(),
        template.context_after.copy(),
        file_path,
    )


def generate_negative_context(
    token: str,
    var_name: Optional[str] = None,
    negative_type: str = "safe",
    language: Optional[str] = None,
) -> Tuple[str, List[str], List[str], str]:
    """
    Generate negative (false positive) context for a token.

    Args:
        token: The token (Git SHA, UUID, etc.)
        var_name: Optional variable name (random if not provided)
        negative_type: Type of negative context: 'safe', 'test', 'doc'
        language: Optional language (random if not provided)

    Returns:
        Tuple of (line_content, context_before, context_after, file_path)
    """
    if negative_type == "test":
        if var_name is None:
            var_name = random.choice(TEST_VAR_NAMES)
        templates = TEST_TEMPLATES
    elif negative_type == "doc":
        if var_name is None:
            var_name = random.choice(["example_key", "sample_token", "demo_secret"])
        templates = DOC_TEMPLATES
    else:  # safe
        if var_name is None:
            var_name = random.choice(SAFE_VAR_NAMES)

        # Select templates based on language
        if language == "python":
            templates = PYTHON_SAFE_TEMPLATES
        elif language in ("javascript", "typescript"):
            templates = JS_SAFE_TEMPLATES
        elif language == "yaml":
            templates = YAML_SAFE_TEMPLATES
        elif language == "json":
            templates = JSON_SAFE_TEMPLATES
        elif language == "shell":
            templates = SHELL_SAFE_TEMPLATES
        elif language == "go":
            templates = GO_SAFE_TEMPLATES
        elif language == "java":
            templates = JAVA_SAFE_TEMPLATES
        else:
            all_templates = (
                PYTHON_SAFE_TEMPLATES
                + JS_SAFE_TEMPLATES
                + YAML_SAFE_TEMPLATES
                + JSON_SAFE_TEMPLATES
                + SHELL_SAFE_TEMPLATES
                + GO_SAFE_TEMPLATES
                + JAVA_SAFE_TEMPLATES
            )
            templates = all_templates

    template = random.choice(templates)

    line_content = template.line_template.format(var_name=var_name, token=token)
    file_path = template.file_path_template.format(name=var_name.lower())

    return (
        line_content,
        template.context_before.copy(),
        template.context_after.copy(),
        file_path,
    )


# Unified template pools for label-agnostic context generation
# These combine secret and safe templates to prevent template bias
PYTHON_UNIFIED_TEMPLATES = PYTHON_SECRET_TEMPLATES + PYTHON_SAFE_TEMPLATES
JS_UNIFIED_TEMPLATES = JS_SECRET_TEMPLATES + JS_SAFE_TEMPLATES
YAML_UNIFIED_TEMPLATES = YAML_SECRET_TEMPLATES + YAML_SAFE_TEMPLATES
JSON_UNIFIED_TEMPLATES = JSON_SECRET_TEMPLATES + JSON_SAFE_TEMPLATES
SHELL_UNIFIED_TEMPLATES = SHELL_SECRET_TEMPLATES + SHELL_SAFE_TEMPLATES
GO_UNIFIED_TEMPLATES = GO_SECRET_TEMPLATES + GO_SAFE_TEMPLATES
JAVA_UNIFIED_TEMPLATES = JAVA_SECRET_TEMPLATES + JAVA_SAFE_TEMPLATES
ENV_UNIFIED_TEMPLATES = ENV_SECRET_TEMPLATES  # .env files - no safe variant


def _get_unified_templates(language: str, context_type: str) -> List[ContextTemplate]:
    """
    Get unified templates for a language, optionally filtered by context type.

    This function returns templates from BOTH secret and safe pools
    to prevent template-based shortcut learning.
    """
    # Base templates by language
    if language == "python":
        templates = PYTHON_UNIFIED_TEMPLATES.copy()
    elif language in ("javascript", "typescript"):
        templates = JS_UNIFIED_TEMPLATES.copy()
    elif language == "yaml":
        templates = YAML_UNIFIED_TEMPLATES.copy()
    elif language == "json":
        templates = JSON_UNIFIED_TEMPLATES.copy()
    elif language in ("dotenv", "env"):
        templates = ENV_UNIFIED_TEMPLATES.copy()
    elif language == "shell":
        templates = SHELL_UNIFIED_TEMPLATES.copy()
    elif language == "go":
        templates = GO_UNIFIED_TEMPLATES.copy()
    elif language == "java":
        templates = JAVA_UNIFIED_TEMPLATES.copy()
    else:
        # Fallback: all templates from all languages
        templates = (
            PYTHON_UNIFIED_TEMPLATES
            + JS_UNIFIED_TEMPLATES
            + YAML_UNIFIED_TEMPLATES
            + JSON_UNIFIED_TEMPLATES
            + ENV_UNIFIED_TEMPLATES
            + SHELL_UNIFIED_TEMPLATES
            + GO_UNIFIED_TEMPLATES
            + JAVA_UNIFIED_TEMPLATES
        )

    # Add context-type-specific templates
    if context_type == "test":
        templates = templates + TEST_TEMPLATES
    elif context_type == "documentation":
        templates = templates + DOC_TEMPLATES

    return templates


def generate_context(
    token: str,
    var_name: str,
    language: str,
    context_type: str,
) -> Tuple[str, List[str], List[str], str]:
    """
    Generate label-agnostic context for a token.

    This function selects templates from a unified pool containing BOTH
    secret-like and safe-like templates, preventing the model from
    learning shortcuts based on template patterns.

    Args:
        token: The token to embed in the context
        var_name: Variable name to use in the template
        language: Programming language (python, javascript, yaml, etc.)
        context_type: Context type (production, test, documentation, configuration)

    Returns:
        Tuple of (line_content, context_before, context_after, file_path)
    """
    templates = _get_unified_templates(language, context_type)

    if not templates:
        # Fallback to Python if no templates found
        templates = PYTHON_UNIFIED_TEMPLATES

    template = random.choice(templates)

    # Format the line content with the token and variable name
    line_content = template.line_template.format(var_name=var_name, token=token)

    # Generate file path using the template
    file_path = template.file_path_template.format(name=var_name.lower())

    return (
        line_content,
        template.context_before.copy(),
        template.context_after.copy(),
        file_path,
    )


__all__ = [
    "generate_context",
    "generate_positive_context",
    "generate_negative_context",
    "SECRET_VAR_NAMES",
    "SAFE_VAR_NAMES",
    "TEST_VAR_NAMES",
    "ContextTemplate",
]
