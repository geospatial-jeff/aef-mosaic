//! Pipeline orchestration for chunk processing.

mod chunk_processor;
mod metrics;
mod scheduler;

pub use chunk_processor::{ChunkProcessor, ChunkResult};
pub use metrics::{Metrics, MetricsReporter, MetricsSnapshot};
pub use scheduler::{MetaTile, Scheduler, SchedulerConfig, SchedulerStats};
