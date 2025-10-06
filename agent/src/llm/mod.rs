use crate::Result;
use crate::tools::{ToolCall, ToolDefinition};
use async_trait::async_trait;

mod openai;
pub use openai::OpenAI;

#[derive(Clone)]
pub enum Message {
    User(String),
    Assistant(String, Vec<ToolCall>),
    System(String),
    Tool {
        id: String,
        name: String,
        result: String,
    },
}

pub struct CompletionRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
}

pub struct CompletionResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait LLM {
    async fn completion<'a>(&self, request: CompletionRequest<'a>) -> Result<CompletionResponse>;
}
