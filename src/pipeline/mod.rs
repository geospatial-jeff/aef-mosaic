//! Pipeline orchestration for chunk processing.

mod metrics;
mod stages;

pub use metrics::{Metrics, MetricsReporter, MetricsSnapshot};
pub use stages::{Pipeline, PipelineConfig, PipelineStats};
