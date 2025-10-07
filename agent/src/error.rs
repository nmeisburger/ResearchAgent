use async_openai::error::OpenAIError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Openai error: {0}")]
    OpenaiError(#[from] OpenAIError),

    #[error("No response from llm: {0}")]
    LLMResponseError(String),

    #[error("Tool {0} does not exist")]
    ToolDoesNotExist(String),

    #[error("Missing arg: {0}")]
    MissingArg(String),

    #[error("Task join error: {0}")]
    TaskJoinError(#[from] tokio::task::JoinError),

    #[error("Agent workflow error: {0}")]
    AgentWorkflowError(String),

    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
}
