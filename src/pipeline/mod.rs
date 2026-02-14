//! Pipeline orchestration for chunk processing.

mod chunk_processor;
mod metrics;
mod scheduler;
mod stages;

pub use chunk_processor::{ChunkProcessor, ChunkResult};
pub use metrics::{Metrics, MetricsReporter, MetricsSnapshot};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerStats};
pub use stages::{Pipeline, PipelineConfig, PipelineStats};
