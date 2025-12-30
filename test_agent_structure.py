#!/usr/bin/env python3
"""Quick test script for the refactored agent with @tool decorators."""

import sys
from agent import YouTubeAgent

# Test with a short video
TEST_VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
API_KEY = "test_key"  # Will fail but tests import

try:
    print("Testing agent import and initialization...")
    agent = YouTubeAgent(API_KEY)
    print("✓ Agent initialized successfully")
    print(f"✓ Tools loaded: {len(agent.tools)}")
    print(f"✓ Tool names: {[t.name for t in agent.tools]}")
    print("\nSUCCESS: Agent structure is valid!")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
