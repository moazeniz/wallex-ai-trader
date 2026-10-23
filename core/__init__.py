"""
Core Layer

This module contains the main pipeline:
- Analysis pipeline orchestration
- Signal generation
- Database persistence
- Performance tracking

Note: Uses lazy imports to avoid loading LLM dependencies for lite modules.
"""


def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once."""
    if name == "SignalDatabase":
        from .database import SignalDatabase
        return SignalDatabase
    elif name == "SignalRecord":
        from .database import SignalRecord
        return SignalRecord
    elif name == "create_database":
        from .database import create_database
        return create_database
    elif name == "SignalGenerator":
        from .signal_generator import SignalGenerator
        return SignalGenerator
    elif name == "SignalResult":
        from .signal_generator import SignalResult
        return SignalResult
    elif name == "create_signal_generator":
        from .signal_generator import create_signal_generator
        return create_signal_generator
    elif name == "AnalysisPipeline":
        from .pipeline import AnalysisPipeline
        return AnalysisPipeline
    elif name == "QuickAnalyzer":
        from .pipeline import QuickAnalyzer
        return QuickAnalyzer
    elif name == "create_pipeline":
        from .pipeline import create_pipeline
        return create_pipeline
    elif name == "PerformanceTracker":
        from .performance_tracker import PerformanceTracker
        return PerformanceTracker
    elif name == "PerformanceMetrics":
        from .performance_tracker import PerformanceMetrics
        return PerformanceMetrics
    elif name == "create_performance_tracker":
        from .performance_tracker import create_performance_tracker
        return create_performance_tracker
    elif name == "LiteAnalyzer":
        from .lite_analyzer import LiteAnalyzer
        return LiteAnalyzer
    elif name == "get_lite_analyzer":
        from .lite_analyzer import get_lite_analyzer
        return get_lite_analyzer
    elif name == "LitePipeline":
        from .lite_pipeline import LitePipeline
        return LitePipeline
    elif name == "get_lite_pipeline":
        from .lite_pipeline import get_lite_pipeline
        return get_lite_pipeline
    elif name == "VirtualInvestor":
        from .virtual_investor import VirtualInvestor
        return VirtualInvestor
    elif name == "get_virtual_investor":
        from .virtual_investor import get_virtual_investor
        return get_virtual_investor
    else:
        raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = [
    "SignalDatabase",
    "SignalRecord",
    "create_database",
    "SignalGenerator",
    "SignalResult",
    "create_signal_generator",
    "AnalysisPipeline",
    "QuickAnalyzer",
    "create_pipeline",
    "PerformanceTracker",
    "PerformanceMetrics",
    "create_performance_tracker",
    "LiteAnalyzer",
    "get_lite_analyzer",
    "LitePipeline",
    "get_lite_pipeline",
    "VirtualInvestor",
    "get_virtual_investor",
]
