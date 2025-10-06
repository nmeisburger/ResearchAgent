mod agent;
pub mod callbacks;
mod error;
pub mod llm;
pub mod tools;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub use agent::{Agent, AgentBuilder, StopCondition};
