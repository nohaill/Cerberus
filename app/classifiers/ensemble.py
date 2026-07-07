"""
Runs the full classifier ensemble concurrently.

Classifiers are sync (transformers pipelines aren't async-native), so we run
them in a bounded thread pool and gather results with asyncio -- this is what
keeps the gateway's p99 latency close to the *slowest single classifier*
rather than the *sum* of all classifiers, which matters a lot once you're
running 4-6 models per request.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.classifiers.base import BaseClassifier
from app.classifiers.hf_classifier import (
    PIIClassifier,
    build_ai_text_detector,
    build_emotion_classifier,
    build_prompt_injection_classifier,
)
from app.config import get_settings
from app.schemas import ClassifierResult


class ClassifierEnsemble:
    def __init__(self, classifiers: list[BaseClassifier] | None = None):
        self.classifiers = classifiers or [
            build_prompt_injection_classifier(),
            build_emotion_classifier(),
            build_ai_text_detector(),
            PIIClassifier(),
        ]
        settings = get_settings()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_classifiers)

    def preload(self) -> None:
        """Load all model weights from disk into memory at process startup.

        Runs each classifier's preload() in the thread pool concurrently so
        total startup time is bounded by the slowest single model, not their
        sum. Call this once from the FastAPI lifespan handler -- not lazily
        on first request -- so:
        1. The meta tensor bug (PyTorch 2.x + CPU-only) is caught at startup,
           not silently on the first real request.
        2. First-request latency is not inflated by model loading time.
        3. All workers share the same pre-loaded pipeline objects (one copy
           per model per process, not one per request).
        """
        import concurrent.futures

        def _preload_one(clf):
            if hasattr(clf, "preload"):
                clf.preload()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.classifiers)) as pool:
            futures = [pool.submit(_preload_one, clf) for clf in self.classifiers]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # re-raises any exception so startup fails loudly

    async def run(self, text: str) -> list[ClassifierResult]:
        loop = asyncio.get_event_loop()
        settings = get_settings()
        tasks = [
            loop.run_in_executor(self._executor, clf.predict, text) for clf in self.classifiers
        ]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Collect whatever finished; return timeout errors for the rest.
            # Note: don't call task.exception() on cancelled futures -- it
            # raises CancelledError. Check done() AND not cancelled() first.
            final = []
            for clf, task in zip(self.classifiers, tasks):
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        final.append(ClassifierResult(name=clf.name, label="error", score=0.0, latency_ms=0.0, error=str(exc)))
                    else:
                        final.append(task.result())
                else:
                    final.append(ClassifierResult(
                        name=clf.name, label="timeout", score=0.0,
                        latency_ms=settings.request_timeout_seconds * 1000, error="timeout"
                    ))
            return final

        final: list[ClassifierResult] = []
        for clf, r in zip(self.classifiers, results):
            if isinstance(r, Exception):
                final.append(ClassifierResult(name=clf.name, label="error", score=0.0, latency_ms=0.0, error=str(r)))
            else:
                final.append(r)
        return final

    def get_pii_classifier(self) -> PIIClassifier | None:
        for c in self.classifiers:
            if isinstance(c, PIIClassifier):
                return c
        return None
