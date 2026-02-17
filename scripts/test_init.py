
import sys
import os
import asyncio
import time
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Orchestrator initialization test...")
    start_time = time.time()
    
    try:
        orch = Orchestrator(session_id="test-session")
        logger.info(f"Orchestrator created in {time.time() - start_time:.2f}s")
        
        step_start = time.time()
        await orch.initialize()
        logger.info(f"Orchestrator initialized in {time.time() - step_start:.2f}s")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
